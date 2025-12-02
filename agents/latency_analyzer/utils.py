# utils.py - Analysis tools for latency analyzer agent
import os
import json
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional, Dict, List, Any
import pandas as pd
import numpy as np
from google.cloud import bigquery
from dotenv import load_dotenv

load_dotenv()

PROJECT_ID = os.getenv('PROJECT_ID')
DATASET = os.getenv('DATASET', 'gemini_logs')
GEMINI_LOG_TABLE = os.getenv('GEMINI_LOG_TABLE', 'gemini_logs')

# Custom JSON encoder to handle Decimal and datetime types
class AnalysisEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        if isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if pd.isna(obj):
            return None
        return super(AnalysisEncoder, self).default(obj)


def parse_time_range(time_range: str) -> tuple[str, str]:
    """
    Parse time range string into start and end timestamps.
    
    Formats:
    - "24h" -> last 24 hours
    - "7d" -> last 7 days
    - "30d" -> last 30 days
    - "YYYY-MM-DD to YYYY-MM-DD" -> custom range
    """
    now = datetime.utcnow()
    
    if time_range.endswith('h'):
        hours = int(time_range[:-1])
        start = now - timedelta(hours=hours)
        end = now
    elif time_range.endswith('d'):
        days = int(time_range[:-1])
        start = now - timedelta(days=days)
        end = now
    elif ' to ' in time_range:
        start_str, end_str = time_range.split(' to ')
        start = datetime.strptime(start_str.strip(), '%Y-%m-%d')
        end = datetime.strptime(end_str.strip(), '%Y-%m-%d') + timedelta(days=1)
    else:
        # Default to last 24 hours
        start = now - timedelta(hours=24)
        end = now
    
    return start.strftime('%Y-%m-%d %H:%M:%S'), end.strftime('%Y-%m-%d %H:%M:%S')


def execute_bigquery(query: str) -> pd.DataFrame:
    """Execute BigQuery query and return DataFrame."""
    if not PROJECT_ID:
        raise ValueError("PROJECT_ID environment variable is not set")
    
    client = bigquery.Client(project=PROJECT_ID)
    return client.query(query).to_dataframe()


def get_overall_statistics(
    time_range: str = "24h",
    model_name: Optional[str] = None,
    agent_name: Optional[str] = None
) -> str:
    """
    Get overall latency and token statistics.
    
    Args:
        time_range: Time range to analyze (e.g., "24h", "7d", "2025-01-01 to 2025-01-31")
        model_name: Filter by specific model (optional)
        agent_name: Filter by specific agent (optional)
        
    Returns:
        JSON string with overall statistics
    """
    try:
        start_time, end_time = parse_time_range(time_range)
        
        # Build WHERE clause
        where_clauses = [
            f"T.logging_time BETWEEN '{start_time}' AND '{end_time}'",
            "T.full_request IS NOT NULL",
            "T.full_response IS NOT NULL",
            "T.model IS NOT NULL",
            "JSON_VALUE(T.metadata.request_latency) IS NOT NULL"
        ]
        
        if model_name:
            where_clauses.append(f"T.model LIKE '%{model_name}%'")
        if agent_name:
            where_clauses.append(f"JSON_VALUE(T.full_request.labels.adk_agent_name) = '{agent_name}'")
        
        where_clause = " AND ".join(where_clauses)
        
        query = f"""
        SELECT
          COUNT(*) as total_requests,
          AVG(CAST(JSON_EXTRACT_SCALAR(T.metadata, '$.request_latency') AS FLOAT64) / 1000) AS mean_latency,
          APPROX_QUANTILES(CAST(JSON_EXTRACT_SCALAR(T.metadata, '$.request_latency') AS FLOAT64) / 1000, 100)[OFFSET(50)] AS median_latency,
          APPROX_QUANTILES(CAST(JSON_EXTRACT_SCALAR(T.metadata, '$.request_latency') AS FLOAT64) / 1000, 100)[OFFSET(90)] AS p90_latency,
          APPROX_QUANTILES(CAST(JSON_EXTRACT_SCALAR(T.metadata, '$.request_latency') AS FLOAT64) / 1000, 100)[OFFSET(95)] AS p95_latency,
          APPROX_QUANTILES(CAST(JSON_EXTRACT_SCALAR(T.metadata, '$.request_latency') AS FLOAT64) / 1000, 100)[OFFSET(99)] AS p99_latency,
          MIN(CAST(JSON_EXTRACT_SCALAR(T.metadata, '$.request_latency') AS FLOAT64) / 1000) AS min_latency,
          MAX(CAST(JSON_EXTRACT_SCALAR(T.metadata, '$.request_latency') AS FLOAT64) / 1000) AS max_latency,
          STDDEV(CAST(JSON_EXTRACT_SCALAR(T.metadata, '$.request_latency') AS FLOAT64) / 1000) AS std_latency,
          AVG(SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.promptTokenCount) AS INT64)) AS mean_input_tokens,
          AVG(SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.candidatesTokenCount) AS INT64)) AS mean_output_tokens,
          AVG(SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.thoughtsTokenCount) AS INT64)) AS mean_thought_tokens,
          AVG(SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.totalTokenCount) AS INT64)) AS mean_total_tokens,
          SUM(SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.totalTokenCount) AS INT64)) AS total_tokens
        FROM
          `{PROJECT_ID}.{DATASET}.{GEMINI_LOG_TABLE}` AS T
        WHERE
          {where_clause}
        """
        
        df = execute_bigquery(query)
        
        if df.empty or df.iloc[0]['total_requests'] == 0:
            return json.dumps({
                "error": "No data found for the specified criteria",
                "metadata": {
                    "time_range": f"{start_time} to {end_time}",
                    "model_name": model_name,
                    "agent_name": agent_name
                }
            })
        
        row = df.iloc[0]
        
        result = {
            "metadata": {
                "time_range": f"{start_time} to {end_time}",
                "model_name": model_name or "all",
                "agent_name": agent_name or "all",
                "total_requests": int(row['total_requests'])
            },
            "latency": {
                "mean": float(row['mean_latency']) if pd.notna(row['mean_latency']) else None,
                "median": float(row['median_latency']) if pd.notna(row['median_latency']) else None,
                "p90": float(row['p90_latency']) if pd.notna(row['p90_latency']) else None,
                "p95": float(row['p95_latency']) if pd.notna(row['p95_latency']) else None,
                "p99": float(row['p99_latency']) if pd.notna(row['p99_latency']) else None,
                "min": float(row['min_latency']) if pd.notna(row['min_latency']) else None,
                "max": float(row['max_latency']) if pd.notna(row['max_latency']) else None,
                "std": float(row['std_latency']) if pd.notna(row['std_latency']) else None
            },
            "tokens": {
                "mean_input": float(row['mean_input_tokens']) if pd.notna(row['mean_input_tokens']) else None,
                "mean_output": float(row['mean_output_tokens']) if pd.notna(row['mean_output_tokens']) else None,
                "mean_thought": float(row['mean_thought_tokens']) if pd.notna(row['mean_thought_tokens']) else None,
                "mean_total": float(row['mean_total_tokens']) if pd.notna(row['mean_total_tokens']) else None,
                "total": int(row['total_tokens']) if pd.notna(row['total_tokens']) else None
            },
            "summary": f"Analyzed {int(row['total_requests'])} requests. Mean latency: {float(row['mean_latency']):.2f}s, P95: {float(row['p95_latency']):.2f}s"
        }
        
        return json.dumps(result, cls=AnalysisEncoder)
        
    except Exception as e:
        logging.error(f"Error in get_overall_statistics: {str(e)}")
        return json.dumps({"error": str(e)})


def get_latency_distribution(
    time_range: str = "24h",
    model_name: Optional[str] = None,
    agent_name: Optional[str] = None
) -> str:
    """
    Get latency distribution categorized into buckets.
    
    Returns histogram data showing how many requests fall into each latency category.
    """
    try:
        start_time, end_time = parse_time_range(time_range)
        
        where_clauses = [
            f"T.logging_time BETWEEN '{start_time}' AND '{end_time}'",
            "T.full_request IS NOT NULL",
            "T.full_response IS NOT NULL",
            "JSON_VALUE(T.metadata.request_latency) IS NOT NULL"
        ]
        
        if model_name:
            where_clauses.append(f"T.model LIKE '%{model_name}%'")
        if agent_name:
            where_clauses.append(f"JSON_VALUE(T.full_request.labels.adk_agent_name) = '{agent_name}'")
        
        where_clause = " AND ".join(where_clauses)
        
        query = f"""
        WITH latency_data AS (
          SELECT
            CAST(JSON_EXTRACT_SCALAR(T.metadata, '$.request_latency') AS FLOAT64) / 1000 AS latency_seconds
          FROM
            `{PROJECT_ID}.{DATASET}.{GEMINI_LOG_TABLE}` AS T
          WHERE
            {where_clause}
        )
        SELECT
          CASE
            WHEN latency_seconds < 1.0 THEN 'Fast (<1s)'
            WHEN latency_seconds < 2.0 THEN 'Medium (1-2s)'
            WHEN latency_seconds < 3.0 THEN 'Slow (2-3s)'
            WHEN latency_seconds < 5.0 THEN 'Very Slow (3-5s)'
            ELSE 'Outliers (5s+)'
          END AS category,
          COUNT(*) as count,
          AVG(latency_seconds) as avg_latency,
          MIN(latency_seconds) as min_latency,
          MAX(latency_seconds) as max_latency
        FROM latency_data
        GROUP BY category
        ORDER BY 
          CASE category
            WHEN 'Fast (<1s)' THEN 1
            WHEN 'Medium (1-2s)' THEN 2
            WHEN 'Slow (2-3s)' THEN 3
            WHEN 'Very Slow (3-5s)' THEN 4
            WHEN 'Outliers (5s+)' THEN 5
          END
        """
        
        df = execute_bigquery(query)
        
        if df.empty:
            return json.dumps({
                "error": "No data found",
                "metadata": {"time_range": f"{start_time} to {end_time}"}
            })
        
        total_requests = df['count'].sum()
        distribution = []
        
        for _, row in df.iterrows():
            distribution.append({
                "category": row['category'],
                "count": int(row['count']),
                "percentage": float(row['count'] / total_requests * 100),
                "avg_latency": float(row['avg_latency']),
                "min_latency": float(row['min_latency']),
                "max_latency": float(row['max_latency'])
            })
        
        result = {
            "metadata": {
                "time_range": f"{start_time} to {end_time}",
                "total_requests": int(total_requests)
            },
            "distribution": distribution,
            "summary": f"Distribution: {distribution[0]['category']} has {distribution[0]['percentage']:.1f}% of requests"
        }
        
        return json.dumps(result, cls=AnalysisEncoder)
        
    except Exception as e:
        logging.error(f"Error in get_latency_distribution: {str(e)}")
        return json.dumps({"error": str(e)})


def get_hourly_patterns(
    time_range: str = "24h",
    model_name: Optional[str] = None,
    agent_name: Optional[str] = None
) -> str:
    """
    Get hourly latency patterns including working vs weekend comparison.
    
    Returns hourly averages, request counts, and identifies peak hours.
    """
    try:
        start_time, end_time = parse_time_range(time_range)
        
        where_clauses = [
            f"T.logging_time BETWEEN '{start_time}' AND '{end_time}'",
            "T.full_request IS NOT NULL",
            "T.full_response IS NOT NULL",
            "JSON_VALUE(T.metadata.request_latency) IS NOT NULL"
        ]
        
        if model_name:
            where_clauses.append(f"T.model LIKE '%{model_name}%'")
        if agent_name:
            where_clauses.append(f"JSON_VALUE(T.full_request.labels.adk_agent_name) = '{agent_name}'")
        
        where_clause = " AND ".join(where_clauses)
        
        query = f"""
        SELECT
          EXTRACT(HOUR FROM T.logging_time) AS hour,
          EXTRACT(DAYOFWEEK FROM T.logging_time) AS day_of_week,
          CASE WHEN EXTRACT(DAYOFWEEK FROM T.logging_time) IN (1, 7) THEN 'weekend' ELSE 'working' END AS day_type,
          COUNT(*) as request_count,
          AVG(CAST(JSON_EXTRACT_SCALAR(T.metadata, '$.request_latency') AS FLOAT64) / 1000) AS avg_latency,
          MIN(CAST(JSON_EXTRACT_SCALAR(T.metadata, '$.request_latency') AS FLOAT64) / 1000) AS min_latency,
          MAX(CAST(JSON_EXTRACT_SCALAR(T.metadata, '$.request_latency') AS FLOAT64) / 1000) AS max_latency
        FROM
          `{PROJECT_ID}.{DATASET}.{GEMINI_LOG_TABLE}` AS T
        WHERE
          {where_clause}
        GROUP BY hour, day_of_week, day_type
        ORDER BY hour
        """
        
        df = execute_bigquery(query)
        
        if df.empty:
            return json.dumps({"error": "No data found"})
        
        # Aggregate by hour and day type
        hourly_working = df[df['day_type'] == 'working'].groupby('hour').agg({
            'request_count': 'sum',
            'avg_latency': 'mean'
        }).reset_index()
        
        hourly_weekend = df[df['day_type'] == 'weekend'].groupby('hour').agg({
            'request_count': 'sum',
            'avg_latency': 'mean'
        }).reset_index()
        
        working_hours = []
        for _, row in hourly_working.iterrows():
            working_hours.append({
                "hour": int(row['hour']),
                "request_count": int(row['request_count']),
                "avg_latency": float(row['avg_latency'])
            })
        
        weekend_hours = []
        for _, row in hourly_weekend.iterrows():
            weekend_hours.append({
                "hour": int(row['hour']),
                "request_count": int(row['request_count']),
                "avg_latency": float(row['avg_latency'])
            })
        
        # Find peak hours
        if not hourly_working.empty:
            peak_hour_working = hourly_working.loc[hourly_working['request_count'].idxmax()]
            slowest_hour_working = hourly_working.loc[hourly_working['avg_latency'].idxmax()]
        else:
            peak_hour_working = None
            slowest_hour_working = None
        
        result = {
            "metadata": {
                "time_range": f"{start_time} to {end_time}",
                "total_requests": int(df['request_count'].sum())
            },
            "working_days": working_hours,
            "weekend_days": weekend_hours,
            "insights": {
                "peak_hour": int(peak_hour_working['hour']) if peak_hour_working is not None else None,
                "peak_hour_requests": int(peak_hour_working['request_count']) if peak_hour_working is not None else None,
                "slowest_hour": int(slowest_hour_working['hour']) if slowest_hour_working is not None else None,
                "slowest_hour_latency": float(slowest_hour_working['avg_latency']) if slowest_hour_working is not None else None
            },
            "summary": f"Peak hour: {int(peak_hour_working['hour'])}:00 with {int(peak_hour_working['request_count'])} requests" if peak_hour_working is not None else "No data"
        }
        
        return json.dumps(result, cls=AnalysisEncoder)
        
    except Exception as e:
        logging.error(f"Error in get_hourly_patterns: {str(e)}")
        return json.dumps({"error": str(e)})


def get_agent_comparison(
    time_range: str = "24h",
    model_name: Optional[str] = None
) -> str:
    """
    Compare performance across different agents.
    
    Returns per-agent statistics including calls, latency, and token usage.
    """
    try:
        start_time, end_time = parse_time_range(time_range)
        
        where_clauses = [
            f"T.logging_time BETWEEN '{start_time}' AND '{end_time}'",
            "T.full_request IS NOT NULL",
            "T.full_response IS NOT NULL",
            "JSON_VALUE(T.metadata.request_latency) IS NOT NULL"
        ]
        
        if model_name:
            where_clauses.append(f"T.model LIKE '%{model_name}%'")
        
        where_clause = " AND ".join(where_clauses)
        
        query = f"""
        SELECT
          COALESCE(JSON_VALUE(T.full_request.labels.adk_agent_name), 'unknown') AS agent_name,
          COUNT(*) as total_calls,
          AVG(CAST(JSON_EXTRACT_SCALAR(T.metadata, '$.request_latency') AS FLOAT64) / 1000) AS avg_latency,
          APPROX_QUANTILES(CAST(JSON_EXTRACT_SCALAR(T.metadata, '$.request_latency') AS FLOAT64) / 1000, 100)[OFFSET(95)] AS p95_latency,
          AVG(SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.promptTokenCount) AS INT64)) AS avg_input_tokens,
          AVG(SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.candidatesTokenCount) AS INT64)) AS avg_output_tokens,
          SUM(SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.totalTokenCount) AS INT64)) AS total_tokens
        FROM
          `{PROJECT_ID}.{DATASET}.{GEMINI_LOG_TABLE}` AS T
        WHERE
          {where_clause}
        GROUP BY agent_name
        ORDER BY total_calls DESC
        """
        
        df = execute_bigquery(query)
        
        if df.empty:
            return json.dumps({"error": "No data found"})
        
        agents = []
        for _, row in df.iterrows():
            # Calculate efficiency score (lower is better): latency per 1000 tokens
            efficiency = None
            if pd.notna(row['avg_input_tokens']) and row['avg_input_tokens'] > 0:
                efficiency = float(row['avg_latency']) / (float(row['avg_input_tokens']) / 1000)
            
            agents.append({
                "agent_name": row['agent_name'],
                "total_calls": int(row['total_calls']),
                "avg_latency": float(row['avg_latency']),
                "p95_latency": float(row['p95_latency']) if pd.notna(row['p95_latency']) else None,
                "avg_input_tokens": float(row['avg_input_tokens']) if pd.notna(row['avg_input_tokens']) else None,
                "avg_output_tokens": float(row['avg_output_tokens']) if pd.notna(row['avg_output_tokens']) else None,
                "total_tokens": int(row['total_tokens']) if pd.notna(row['total_tokens']) else None,
                "efficiency_score": efficiency
            })
        
        # Rank by efficiency
        agents_with_efficiency = [a for a in agents if a['efficiency_score'] is not None]
        if agents_with_efficiency:
            agents_with_efficiency.sort(key=lambda x: x['efficiency_score'])
            best_agent = agents_with_efficiency[0]['agent_name']
            worst_agent = agents_with_efficiency[-1]['agent_name']
        else:
            best_agent = None
            worst_agent = None
        
        result = {
            "metadata": {
                "time_range": f"{start_time} to {end_time}",
                "total_agents": len(agents)
            },
            "agents": agents,
            "insights": {
                "most_efficient_agent": best_agent,
                "least_efficient_agent": worst_agent,
                "most_active_agent": agents[0]['agent_name'] if agents else None
            },
            "summary": f"Analyzed {len(agents)} agents. Most active: {agents[0]['agent_name']} with {agents[0]['total_calls']} calls" if agents else "No data"
        }
        
        return json.dumps(result, cls=AnalysisEncoder)
        
    except Exception as e:
        logging.error(f"Error in get_agent_comparison: {str(e)}")
        return json.dumps({"error": str(e)})


def get_token_correlation(
    time_range: str = "24h",
    model_name: Optional[str] = None,
    agent_name: Optional[str] = None
) -> str:
    """
    Analyze correlation between latency and token counts.
    
    Returns correlation coefficients and scatter plot data.
    """
    try:
        start_time, end_time = parse_time_range(time_range)
        
        where_clauses = [
            f"T.logging_time BETWEEN '{start_time}' AND '{end_time}'",
            "T.full_request IS NOT NULL",
            "T.full_response IS NOT NULL",
            "JSON_VALUE(T.metadata.request_latency) IS NOT NULL",
            "SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.promptTokenCount) AS INT64) IS NOT NULL"
        ]
        
        if model_name:
            where_clauses.append(f"T.model LIKE '%{model_name}%'")
        if agent_name:
            where_clauses.append(f"JSON_VALUE(T.full_request.labels.adk_agent_name) = '{agent_name}'")
        
        where_clause = " AND ".join(where_clauses)
        
        query = f"""
        SELECT
          CAST(JSON_EXTRACT_SCALAR(T.metadata, '$.request_latency') AS FLOAT64) / 1000 AS latency,
          SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.promptTokenCount) AS INT64) AS input_tokens,
          SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.candidatesTokenCount) AS INT64) AS output_tokens,
          SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.thoughtsTokenCount) AS INT64) AS thought_tokens
        FROM
          `{PROJECT_ID}.{DATASET}.{GEMINI_LOG_TABLE}` AS T
        WHERE
          {where_clause}
        LIMIT 1000
        """
        
        df = execute_bigquery(query)
        
        if df.empty or len(df) < 2:
            return json.dumps({"error": "Insufficient data for correlation analysis"})
        
        # Calculate correlations
        corr_input = df['latency'].corr(df['input_tokens']) if df['input_tokens'].notna().sum() > 1 else None
        corr_output = df['latency'].corr(df['output_tokens']) if df['output_tokens'].notna().sum() > 1 else None
        corr_thought = df['latency'].corr(df['thought_tokens']) if df['thought_tokens'].notna().sum() > 1 else None
        
        # Sample data points for visualization (limit to 100 for JSON size)
        sample_df = df.sample(min(100, len(df)))
        scatter_data = []
        for _, row in sample_df.iterrows():
            scatter_data.append({
                "latency": float(row['latency']),
                "input_tokens": int(row['input_tokens']) if pd.notna(row['input_tokens']) else None,
                "output_tokens": int(row['output_tokens']) if pd.notna(row['output_tokens']) else None,
                "thought_tokens": int(row['thought_tokens']) if pd.notna(row['thought_tokens']) else None
            })
        
        result = {
            "metadata": {
                "time_range": f"{start_time} to {end_time}",
                "sample_size": len(df)
            },
            "correlations": {
                "latency_vs_input_tokens": float(corr_input) if corr_input is not None else None,
                "latency_vs_output_tokens": float(corr_output) if corr_output is not None else None,
                "latency_vs_thought_tokens": float(corr_thought) if corr_thought is not None else None
            },
            "scatter_data": scatter_data,
            "summary": f"Input tokens correlation: {corr_input:.3f}, Output tokens correlation: {corr_output:.3f}" if corr_input and corr_output else "Insufficient data"
        }
        
        return json.dumps(result, cls=AnalysisEncoder)
        
    except Exception as e:
        logging.error(f"Error in get_token_correlation: {str(e)}")
        return json.dumps({"error": str(e)})


def get_outlier_analysis(
    time_range: str = "24h",
    model_name: Optional[str] = None,
    threshold_std: float = 3.0
) -> str:
    """
    Identify outlier requests (latency > threshold * std dev).
    
    Returns list of outlier requests with full details.
    """
    try:
        start_time, end_time = parse_time_range(time_range)
        
        where_clauses = [
            f"T.logging_time BETWEEN '{start_time}' AND '{end_time}'",
            "T.full_request IS NOT NULL",
            "T.full_response IS NOT NULL",
            "JSON_VALUE(T.metadata.request_latency) IS NOT NULL"
        ]
        
        if model_name:
            where_clauses.append(f"T.model LIKE '%{model_name}%'")
        
        where_clause = " AND ".join(where_clauses)
        
        # First get mean and std
        stats_query = f"""
        SELECT
          AVG(CAST(JSON_EXTRACT_SCALAR(T.metadata, '$.request_latency') AS FLOAT64) / 1000) AS mean_latency,
          STDDEV(CAST(JSON_EXTRACT_SCALAR(T.metadata, '$.request_latency') AS FLOAT64) / 1000) AS std_latency
        FROM
          `{PROJECT_ID}.{DATASET}.{GEMINI_LOG_TABLE}` AS T
        WHERE
          {where_clause}
        """
        
        stats_df = execute_bigquery(stats_query)
        mean_latency = float(stats_df.iloc[0]['mean_latency'])
        std_latency = float(stats_df.iloc[0]['std_latency'])
        threshold = mean_latency + (threshold_std * std_latency)
        
        # Now get outliers
        outliers_query = f"""
        SELECT
          CAST(T.request_id AS STRING) AS request_id,
          T.logging_time,
          CAST(JSON_EXTRACT_SCALAR(T.metadata, '$.request_latency') AS FLOAT64) / 1000 AS latency,
          COALESCE(JSON_VALUE(T.full_request.labels.adk_agent_name), 'unknown') AS agent_name,
          SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.promptTokenCount) AS INT64) AS input_tokens,
          SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.candidatesTokenCount) AS INT64) AS output_tokens,
          SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.thoughtsTokenCount) AS INT64) AS thought_tokens
        FROM
          `{PROJECT_ID}.{DATASET}.{GEMINI_LOG_TABLE}` AS T
        WHERE
          {where_clause}
          AND CAST(JSON_EXTRACT_SCALAR(T.metadata, '$.request_latency') AS FLOAT64) / 1000 > {threshold}
        ORDER BY latency DESC
        LIMIT 50
        """
        
        df = execute_bigquery(outliers_query)
        
        outliers = []
        for _, row in df.iterrows():
            outliers.append({
                "request_id": row['request_id'],
                "timestamp": row['logging_time'].isoformat() if pd.notna(row['logging_time']) else None,
                "latency": float(row['latency']),
                "agent_name": row['agent_name'],
                "input_tokens": int(row['input_tokens']) if pd.notna(row['input_tokens']) else None,
                "output_tokens": int(row['output_tokens']) if pd.notna(row['output_tokens']) else None,
                "thought_tokens": int(row['thought_tokens']) if pd.notna(row['thought_tokens']) else None,
                "std_deviations_above_mean": float((row['latency'] - mean_latency) / std_latency) if std_latency > 0 else None
            })
        
        result = {
            "metadata": {
                "time_range": f"{start_time} to {end_time}",
                "mean_latency": mean_latency,
                "std_latency": std_latency,
                "threshold": threshold,
                "outlier_count": len(outliers)
            },
            "outliers": outliers,
            "summary": f"Found {len(outliers)} outliers with latency > {threshold:.2f}s (mean + {threshold_std} std)"
        }
        
        return json.dumps(result, cls=AnalysisEncoder)
        
    except Exception as e:
        logging.error(f"Error in get_outlier_analysis: {str(e)}")
        return json.dumps({"error": str(e)})


def get_slowest_queries(
    num_queries: int = 10,
    time_range: str = "24h",
    model_name: Optional[str] = None
) -> str:
    """
    Get the N slowest queries with full details for deep analysis.
    
    This is the merged functionality from slow_query_analyzer.
    Returns request IDs and metadata for the slowest queries.
    """
    try:
        start_time, end_time = parse_time_range(time_range)
        
        where_clauses = [
            f"T.logging_time BETWEEN '{start_time}' AND '{end_time}'",
            "T.full_request IS NOT NULL",
            "T.full_response IS NOT NULL",
            "JSON_VALUE(T.metadata.request_latency) IS NOT NULL"
        ]
        
        if model_name:
            where_clauses.append(f"T.model LIKE '%{model_name}%'")
        
        where_clause = " AND ".join(where_clauses)
        
        query = f"""
        SELECT
          CAST(T.request_id AS STRING) AS request_id,
          T.logging_time,
          CAST(JSON_EXTRACT_SCALAR(T.metadata, '$.request_latency') AS FLOAT64) / 1000 AS latency,
          T.model,
          COALESCE(JSON_VALUE(T.full_request.labels.adk_agent_name), 'unknown') AS agent_name,
          SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.promptTokenCount) AS INT64) AS input_tokens,
          SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.candidatesTokenCount) AS INT64) AS output_tokens,
          SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.thoughtsTokenCount) AS INT64) AS thought_tokens,
          SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.totalTokenCount) AS INT64) AS total_tokens
        FROM
          `{PROJECT_ID}.{DATASET}.{GEMINI_LOG_TABLE}` AS T
        WHERE
          {where_clause}
        ORDER BY latency DESC
        LIMIT {num_queries}
        """
        
        df = execute_bigquery(query)
        
        if df.empty:
            return json.dumps({"error": "No data found"})
        
        queries = []
        for _, row in df.iterrows():
            queries.append({
                "request_id": row['request_id'],
                "timestamp": row['logging_time'].isoformat() if pd.notna(row['logging_time']) else None,
                "latency": float(row['latency']),
                "model": row['model'],
                "agent_name": row['agent_name'],
                "input_tokens": int(row['input_tokens']) if pd.notna(row['input_tokens']) else None,
                "output_tokens": int(row['output_tokens']) if pd.notna(row['output_tokens']) else None,
                "thought_tokens": int(row['thought_tokens']) if pd.notna(row['thought_tokens']) else None,
                "total_tokens": int(row['total_tokens']) if pd.notna(row['total_tokens']) else None
            })
        
        result = {
            "metadata": {
                "time_range": f"{start_time} to {end_time}",
                "num_queries": len(queries)
            },
            "slowest_queries": queries,
            "summary": f"Top {len(queries)} slowest queries. Slowest: {queries[0]['latency']:.2f}s with {queries[0]['total_tokens']} tokens"
        }
        
        return json.dumps(result, cls=AnalysisEncoder)
        
    except Exception as e:
        logging.error(f"Error in get_slowest_queries: {str(e)}")
        return json.dumps({"error": str(e)})


def get_query_details(request_id: str) -> str:
    """
    Get full details for a specific query by request_id.
    
    This allows deep-dive analysis of individual slow queries.
    Merged from slow_query_analyzer's fetch_single_query.
    """
    try:
        query = f"""
        SELECT
          T.logging_time,
          CAST(T.request_id AS STRING) AS request_id,
          T.full_request,
          T.full_response,
          T.model,
          JSON_VALUE(T.full_request.labels.adk_agent_name) AS agent_name,
          CAST(JSON_EXTRACT_SCALAR(T.metadata, '$.request_latency') AS FLOAT64) / 1000 AS latency,
          SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.thoughtsTokenCount) AS INT64) AS thought_tokens,
          SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.candidatesTokenCount) AS INT64) AS output_tokens,
          SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.promptTokenCount) AS INT64) AS input_tokens,
          SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.totalTokenCount) AS INT64) AS total_tokens
        FROM
          `{PROJECT_ID}.{DATASET}.{GEMINI_LOG_TABLE}` AS T
        WHERE
          CAST(T.request_id AS STRING) = @request_id
        LIMIT 1
        """
        
        client = bigquery.Client(project=PROJECT_ID)
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("request_id", "STRING", str(request_id))
            ]
        )
        
        df = client.query(query, job_config=job_config).to_dataframe()
        
        if df.empty:
            return json.dumps({"error": f"No record found for request_id: {request_id}"})
        
        row = df.iloc[0]
        
        result = {
            "request_id": row['request_id'],
            "timestamp": row['logging_time'].isoformat() if pd.notna(row['logging_time']) else None,
            "latency": float(row['latency']),
            "model": row['model'],
            "agent_name": row['agent_name'] if pd.notna(row['agent_name']) else 'unknown',
            "tokens": {
                "input": int(row['input_tokens']) if pd.notna(row['input_tokens']) else None,
                "output": int(row['output_tokens']) if pd.notna(row['output_tokens']) else None,
                "thought": int(row['thought_tokens']) if pd.notna(row['thought_tokens']) else None,
                "total": int(row['total_tokens']) if pd.notna(row['total_tokens']) else None
            },
            "full_request": json.loads(row['full_request']) if pd.notna(row['full_request']) else None,
            "full_response": json.loads(row['full_response']) if pd.notna(row['full_response']) else None
        }
        
        return json.dumps(result, cls=AnalysisEncoder, default=str)
        
    except Exception as e:
        logging.error(f"Error in get_query_details: {str(e)}")
        return json.dumps({"error": str(e)})


def get_concurrent_request_impact(
    time_range: str = "24h",
    model_name: Optional[str] = None,
    bucket_size: int = 300
) -> str:
    """
    Analyze impact of concurrent requests on latency.
    
    Groups requests into time buckets and correlates concurrency with latency.
    """
    try:
        start_time, end_time = parse_time_range(time_range)
        
        where_clauses = [
            f"T.logging_time BETWEEN '{start_time}' AND '{end_time}'",
            "T.full_request IS NOT NULL",
            "T.full_response IS NOT NULL",
            "JSON_VALUE(T.metadata.request_latency) IS NOT NULL"
        ]
        
        if model_name:
            where_clauses.append(f"T.model LIKE '%{model_name}%'")
        
        where_clause = " AND ".join(where_clauses)
        
        query = f"""
        WITH bucketed_requests AS (
          SELECT
            TIMESTAMP_TRUNC(T.logging_time, SECOND, 'UTC') AS bucket_start,
            TIMESTAMP_ADD(TIMESTAMP_TRUNC(T.logging_time, SECOND, 'UTC'), INTERVAL {bucket_size} SECOND) AS bucket_end,
            CAST(JSON_EXTRACT_SCALAR(T.metadata, '$.request_latency') AS FLOAT64) / 1000 AS latency
          FROM
            `{PROJECT_ID}.{DATASET}.{GEMINI_LOG_TABLE}` AS T
          WHERE
            {where_clause}
        )
        SELECT
          bucket_start,
          COUNT(*) as concurrent_requests,
          AVG(latency) as avg_latency,
          MIN(latency) as min_latency,
          MAX(latency) as max_latency
        FROM bucketed_requests
        GROUP BY bucket_start
        HAVING COUNT(*) > 0
        ORDER BY bucket_start
        """
        
        df = execute_bigquery(query)
        
        if df.empty or len(df) < 2:
            return json.dumps({"error": "Insufficient data for concurrency analysis"})
        
        # Calculate correlation
        correlation = df['concurrent_requests'].corr(df['avg_latency'])
        
        # Find high concurrency periods
        high_concurrency_threshold = df['concurrent_requests'].quantile(0.9)
        high_concurrency_df = df[df['concurrent_requests'] >= high_concurrency_threshold]
        
        buckets = []
        for _, row in df.head(20).iterrows():  # Limit to 20 buckets for JSON size
            buckets.append({
                "bucket_start": row['bucket_start'].isoformat(),
                "concurrent_requests": int(row['concurrent_requests']),
                "avg_latency": float(row['avg_latency']),
                "min_latency": float(row['min_latency']),
                "max_latency": float(row['max_latency'])
            })
        
        result = {
            "metadata": {
                "time_range": f"{start_time} to {end_time}",
                "bucket_size_seconds": bucket_size,
                "total_buckets": len(df)
            },
            "correlation": {
                "concurrency_vs_latency": float(correlation) if pd.notna(correlation) else None
            },
            "high_concurrency_periods": {
                "threshold": int(high_concurrency_threshold),
                "count": len(high_concurrency_df),
                "avg_latency_during_high_concurrency": float(high_concurrency_df['avg_latency'].mean()) if not high_concurrency_df.empty else None
            },
            "sample_buckets": buckets,
            "summary": f"Concurrency correlation: {correlation:.3f}. High concurrency (>{int(high_concurrency_threshold)} req) has avg latency {float(high_concurrency_df['avg_latency'].mean()):.2f}s" if not high_concurrency_df.empty and pd.notna(correlation) else "Insufficient data"
        }
        
        return json.dumps(result, cls=AnalysisEncoder)
        
    except Exception as e:
        logging.error(f"Error in get_concurrent_request_impact: {str(e)}")
        return json.dumps({"error": str(e)})


def detect_performance_degradation(
    time_range: str = "7d",
    model_name: Optional[str] = None,
    window_size: int = 24
) -> str:
    """
    Detect performance degradation over time using moving averages.
    
    Compares recent performance to baseline to identify trends.
    """
    try:
        start_time, end_time = parse_time_range(time_range)
        
        where_clauses = [
            f"T.logging_time BETWEEN '{start_time}' AND '{end_time}'",
            "T.full_request IS NOT NULL",
            "T.full_response IS NOT NULL",
            "JSON_VALUE(T.metadata.request_latency) IS NOT NULL"
        ]
        
        if model_name:
            where_clauses.append(f"T.model LIKE '%{model_name}%'")
        
        where_clause = " AND ".join(where_clauses)
        
        query = f"""
        SELECT
          TIMESTAMP_TRUNC(T.logging_time, HOUR) AS hour,
          AVG(CAST(JSON_EXTRACT_SCALAR(T.metadata, '$.request_latency') AS FLOAT64) / 1000) AS avg_latency,
          COUNT(*) as request_count
        FROM
          `{PROJECT_ID}.{DATASET}.{GEMINI_LOG_TABLE}` AS T
        WHERE
          {where_clause}
        GROUP BY hour
        ORDER BY hour
        """
        
        df = execute_bigquery(query)
        
        if df.empty or len(df) < window_size:
            return json.dumps({"error": "Insufficient data for trend analysis"})
        
        # Calculate moving average
        df['moving_avg'] = df['avg_latency'].rolling(window=window_size, min_periods=1).mean()
        
        # Compare first quarter vs last quarter
        quarter_size = len(df) // 4
        baseline_latency = df.head(quarter_size)['avg_latency'].mean()
        recent_latency = df.tail(quarter_size)['avg_latency'].mean()
        
        degradation_pct = ((recent_latency - baseline_latency) / baseline_latency * 100) if baseline_latency > 0 else 0
        
        # Detect if trend is increasing
        trend_increasing = recent_latency > baseline_latency
        
        result = {
            "metadata": {
                "time_range": f"{start_time} to {end_time}",
                "window_size_hours": window_size,
                "data_points": len(df)
            },
            "baseline": {
                "latency": float(baseline_latency),
                "period": "first 25% of time range"
            },
            "recent": {
                "latency": float(recent_latency),
                "period": "last 25% of time range"
            },
            "degradation": {
                "percentage_change": float(degradation_pct),
                "is_degrading": trend_increasing and degradation_pct > 5,  # >5% increase
                "severity": "high" if degradation_pct > 20 else "medium" if degradation_pct > 10 else "low" if degradation_pct > 5 else "none"
            },
            "summary": f"Performance {'degraded' if trend_increasing else 'improved'} by {abs(degradation_pct):.1f}%. Recent latency: {recent_latency:.2f}s vs baseline: {baseline_latency:.2f}s"
        }
        
        return json.dumps(result, cls=AnalysisEncoder)
        
    except Exception as e:
        logging.error(f"Error in detect_performance_degradation: {str(e)}")
        return json.dumps({"error": str(e)})


def get_cost_analysis(
    time_range: str = "24h",
    model_name: Optional[str] = None
) -> str:
    """
    Analyze token usage and estimated costs.
    
    Provides cost breakdown by agent and identifies expensive operations.
    """
    try:
        start_time, end_time = parse_time_range(time_range)
        
        where_clauses = [
            f"T.logging_time BETWEEN '{start_time}' AND '{end_time}'",
            "T.full_request IS NOT NULL",
            "T.full_response IS NOT NULL"
        ]
        
        if model_name:
            where_clauses.append(f"T.model LIKE '%{model_name}%'")
        
        where_clause = " AND ".join(where_clauses)
        
        query = f"""
        SELECT
          COALESCE(JSON_VALUE(T.full_request.labels.adk_agent_name), 'unknown') AS agent_name,
          COUNT(*) as total_requests,
          SUM(SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.promptTokenCount) AS INT64)) AS total_input_tokens,
          SUM(SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.candidatesTokenCount) AS INT64)) AS total_output_tokens,
          SUM(SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.totalTokenCount) AS INT64)) AS total_tokens,
          AVG(SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.totalTokenCount) AS INT64)) AS avg_tokens_per_request
        FROM
          `{PROJECT_ID}.{DATASET}.{GEMINI_LOG_TABLE}` AS T
        WHERE
          {where_clause}
        GROUP BY agent_name
        ORDER BY total_tokens DESC
        """
        
        df = execute_bigquery(query)
        
        if df.empty:
            return json.dumps({"error": "No data found"})
        
        # Rough cost estimation (adjust based on actual pricing)
        # Example: $0.075 per 1M input tokens, $0.30 per 1M output tokens for Gemini 1.5 Pro
        INPUT_COST_PER_1M = 0.075
        OUTPUT_COST_PER_1M = 0.30
        
        agents = []
        total_cost = 0
        
        for _, row in df.iterrows():
            input_cost = (float(row['total_input_tokens']) / 1_000_000 * INPUT_COST_PER_1M) if pd.notna(row['total_input_tokens']) else 0
            output_cost = (float(row['total_output_tokens']) / 1_000_000 * OUTPUT_COST_PER_1M) if pd.notna(row['total_output_tokens']) else 0
            agent_cost = input_cost + output_cost
            total_cost += agent_cost
            
            agents.append({
                "agent_name": row['agent_name'],
                "total_requests": int(row['total_requests']),
                "total_tokens": int(row['total_tokens']) if pd.notna(row['total_tokens']) else 0,
                "total_input_tokens": int(row['total_input_tokens']) if pd.notna(row['total_input_tokens']) else 0,
                "total_output_tokens": int(row['total_output_tokens']) if pd.notna(row['total_output_tokens']) else 0,
                "avg_tokens_per_request": float(row['avg_tokens_per_request']) if pd.notna(row['avg_tokens_per_request']) else 0,
                "estimated_cost_usd": float(agent_cost),
                "cost_per_request": float(agent_cost / row['total_requests']) if row['total_requests'] > 0 else 0
            })
        
        result = {
            "metadata": {
                "time_range": f"{start_time} to {end_time}",
                "total_agents": len(agents),
                "pricing_note": "Estimated using Gemini 1.5 Pro pricing. Adjust for actual model."
            },
            "total_cost_usd": float(total_cost),
            "agents": agents,
            "most_expensive_agent": agents[0]['agent_name'] if agents else None,
            "summary": f"Total estimated cost: ${total_cost:.2f}. Most expensive: {agents[0]['agent_name']} (${agents[0]['estimated_cost_usd']:.2f})" if agents else "No data"
        }
        
        return json.dumps(result, cls=AnalysisEncoder)
        
    except Exception as e:
        logging.error(f"Error in get_cost_analysis: {str(e)}")
        return json.dumps({"error": str(e)})


def compare_time_periods(
    period1: str,
    period2: str,
    model_name: Optional[str] = None
) -> str:
    """
    Compare performance between two time periods.
    
    Useful for before/after analysis or A/B testing validation.
    """
    try:
        start1, end1 = parse_time_range(period1)
        start2, end2 = parse_time_range(period2)
        
        def get_period_stats(start, end):
            where_clauses = [
                f"T.logging_time BETWEEN '{start}' AND '{end}'",
                "T.full_request IS NOT NULL",
                "T.full_response IS NOT NULL",
                "JSON_VALUE(T.metadata.request_latency) IS NOT NULL"
            ]
            
            if model_name:
                where_clauses.append(f"T.model LIKE '%{model_name}%'")
            
            where_clause = " AND ".join(where_clauses)
            
            query = f"""
            SELECT
              COUNT(*) as total_requests,
              AVG(CAST(JSON_EXTRACT_SCALAR(T.metadata, '$.request_latency') AS FLOAT64) / 1000) AS mean_latency,
              APPROX_QUANTILES(CAST(JSON_EXTRACT_SCALAR(T.metadata, '$.request_latency') AS FLOAT64) / 1000, 100)[OFFSET(95)] AS p95_latency,
              AVG(SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.totalTokenCount) AS INT64)) AS avg_tokens
            FROM
              `{PROJECT_ID}.{DATASET}.{GEMINI_LOG_TABLE}` AS T
            WHERE
              {where_clause}
            """
            
            df = execute_bigquery(query)
            if df.empty or df.iloc[0]['total_requests'] == 0:
                return None
            
            row = df.iloc[0]
            return {
                "total_requests": int(row['total_requests']),
                "mean_latency": float(row['mean_latency']) if pd.notna(row['mean_latency']) else None,
                "p95_latency": float(row['p95_latency']) if pd.notna(row['p95_latency']) else None,
                "avg_tokens": float(row['avg_tokens']) if pd.notna(row['avg_tokens']) else None
            }
        
        period1_stats = get_period_stats(start1, end1)
        period2_stats = get_period_stats(start2, end2)
        
        if not period1_stats or not period2_stats:
            return json.dumps({"error": "Insufficient data in one or both periods"})
        
        # Calculate differences
        latency_change_pct = ((period2_stats['mean_latency'] - period1_stats['mean_latency']) / period1_stats['mean_latency'] * 100) if period1_stats['mean_latency'] > 0 else 0
        p95_change_pct = ((period2_stats['p95_latency'] - period1_stats['p95_latency']) / period1_stats['p95_latency'] * 100) if period1_stats['p95_latency'] > 0 else 0
        
        result = {
            "period1": {
                "time_range": f"{start1} to {end1}",
                "stats": period1_stats
            },
            "period2": {
                "time_range": f"{start2} to {end2}",
                "stats": period2_stats
            },
            "comparison": {
                "latency_change_percent": float(latency_change_pct),
                "p95_change_percent": float(p95_change_pct),
                "request_count_change": int(period2_stats['total_requests'] - period1_stats['total_requests']),
                "performance_improved": latency_change_pct < 0
            },
            "summary": f"Period 2 vs Period 1: Latency {'improved' if latency_change_pct < 0 else 'degraded'} by {abs(latency_change_pct):.1f}%, P95 {'improved' if p95_change_pct < 0 else 'degraded'} by {abs(p95_change_pct):.1f}%"
        }
        
        return json.dumps(result, cls=AnalysisEncoder)
        
    except Exception as e:
        logging.error(f"Error in compare_time_periods: {str(e)}")
        return json.dumps({"error": str(e)})


def cluster_slow_queries(
    num_queries: int = 50,
    time_range: str = "24h",
    model_name: Optional[str] = None
) -> str:
    """
    Cluster slow queries by similarity and provide breakdown.
    
    Groups queries by common characteristics:
    - Similar latency ranges
    - Similar token patterns (input/output/thought)
    - Same agent
    - Time-based patterns
    
    Returns clusters with representative examples and statistics.
    """
    try:
        start_time, end_time = parse_time_range(time_range)
        
        where_clauses = [
            f"T.logging_time BETWEEN '{start_time}' AND '{end_time}'",
            "T.full_request IS NOT NULL",
            "T.full_response IS NOT NULL",
            "JSON_VALUE(T.metadata.request_latency) IS NOT NULL"
        ]
        
        if model_name:
            where_clauses.append(f"T.model LIKE '%{model_name}%'")
        
        where_clause = " AND ".join(where_clauses)
        
        query = f"""
        SELECT
          CAST(T.request_id AS STRING) AS request_id,
          T.logging_time,
          CAST(JSON_EXTRACT_SCALAR(T.metadata, '$.request_latency') AS FLOAT64) / 1000 AS latency,
          COALESCE(JSON_VALUE(T.full_request.labels.adk_agent_name), 'unknown') AS agent_name,
          SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.promptTokenCount) AS INT64) AS input_tokens,
          SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.candidatesTokenCount) AS INT64) AS output_tokens,
          SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.thoughtsTokenCount) AS INT64) AS thought_tokens,
          SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.totalTokenCount) AS INT64) AS total_tokens,
          -- Extract first 200 chars of request for similarity analysis
          SUBSTR(TO_JSON_STRING(T.full_request), 1, 200) AS request_preview
        FROM
          `{PROJECT_ID}.{DATASET}.{GEMINI_LOG_TABLE}` AS T
        WHERE
          {where_clause}
        ORDER BY latency DESC
        LIMIT {num_queries}
        """
        
        df = execute_bigquery(query)
        
        if df.empty:
            return json.dumps({"error": "No data found"})
        
        # Calculate output + thought tokens
        df['output_thought_tokens'] = df['output_tokens'].fillna(0) + df['thought_tokens'].fillna(0)
        
        # Define clustering criteria
        clusters = {}
        
        # Cluster 1: By latency range
        def get_latency_cluster(latency):
            if latency < 5:
                return "moderate_slow_3-5s"
            elif latency < 10:
                return "slow_5-10s"
            elif latency < 20:
                return "very_slow_10-20s"
            else:
                return "extremely_slow_20s+"
        
        # Cluster 2: By token pattern
        def get_token_cluster(row):
            input_t = row['input_tokens'] or 0
            output_t = row['output_thought_tokens'] or 0
            
            if input_t > 10000:
                return "massive_input_10k+"
            elif output_t > 5000:
                return "massive_output_5k+"
            elif input_t > 5000:
                return "large_input_5k+"
            elif output_t > 2000:
                return "large_output_2k+"
            else:
                return "normal_tokens"
        
        # Group by multiple dimensions
        df['latency_cluster'] = df['latency'].apply(get_latency_cluster)
        df['token_cluster'] = df.apply(get_token_cluster, axis=1)
        
        # Analyze clusters
        cluster_analysis = []
        
        # Group by latency cluster
        for cluster_name, group in df.groupby('latency_cluster'):
            cluster_info = {
                "cluster_type": "latency_range",
                "cluster_name": cluster_name,
                "count": len(group),
                "percentage": float(len(group) / len(df) * 100),
                "avg_latency": float(group['latency'].mean()),
                "avg_input_tokens": float(group['input_tokens'].mean()) if group['input_tokens'].notna().any() else None,
                "avg_output_thought_tokens": float(group['output_thought_tokens'].mean()) if group['output_thought_tokens'].notna().any() else None,
                "top_agents": group['agent_name'].value_counts().head(3).to_dict(),
                "sample_request_ids": group['request_id'].head(3).tolist()
            }
            cluster_analysis.append(cluster_info)
        
        # Group by token pattern
        for cluster_name, group in df.groupby('token_cluster'):
            cluster_info = {
                "cluster_type": "token_pattern",
                "cluster_name": cluster_name,
                "count": len(group),
                "percentage": float(len(group) / len(df) * 100),
                "avg_latency": float(group['latency'].mean()),
                "avg_input_tokens": float(group['input_tokens'].mean()) if group['input_tokens'].notna().any() else None,
                "avg_output_thought_tokens": float(group['output_thought_tokens'].mean()) if group['output_thought_tokens'].notna().any() else None,
                "top_agents": group['agent_name'].value_counts().head(3).to_dict(),
                "sample_request_ids": group['request_id'].head(3).tolist()
            }
            cluster_analysis.append(cluster_info)
        
        # Group by agent
        agent_clusters = []
        for agent_name, group in df.groupby('agent_name'):
            if len(group) >= 3:  # Only include agents with 3+ slow queries
                agent_clusters.append({
                    "agent_name": agent_name,
                    "count": len(group),
                    "percentage": float(len(group) / len(df) * 100),
                    "avg_latency": float(group['latency'].mean()),
                    "avg_input_tokens": float(group['input_tokens'].mean()) if group['input_tokens'].notna().any() else None,
                    "avg_output_thought_tokens": float(group['output_thought_tokens'].mean()) if group['output_thought_tokens'].notna().any() else None
                })
        
        result = {
            "metadata": {
                "time_range": f"{start_time} to {end_time}",
                "total_queries_analyzed": len(df)
            },
            "clusters": cluster_analysis,
            "agent_breakdown": sorted(agent_clusters, key=lambda x: x['count'], reverse=True),
            "insights": {
                "dominant_cluster": max(cluster_analysis, key=lambda x: x['count'])['cluster_name'] if cluster_analysis else None,
                "most_problematic_agent": agent_clusters[0]['agent_name'] if agent_clusters else None
            },
            "summary": f"Analyzed {len(df)} slow queries. Dominant pattern: {max(cluster_analysis, key=lambda x: x['count'])['cluster_name']} ({max(cluster_analysis, key=lambda x: x['count'])['count']} queries)" if cluster_analysis else "No clusters found"
        }
        
        return json.dumps(result, cls=AnalysisEncoder)
        
    except Exception as e:
        logging.error(f"Error in cluster_slow_queries: {str(e)}")
        return json.dumps({"error": str(e)})


def analyze_correlation_detailed(
    time_range: str = "24h",
    model_name: Optional[str] = None
) -> str:
    """
    Detailed correlation analysis including output+thought tokens.
    
    Provides comprehensive correlation matrix and statistical significance.
    """
    try:
        start_time, end_time = parse_time_range(time_range)
        
        where_clauses = [
            f"T.logging_time BETWEEN '{start_time}' AND '{end_time}'",
            "T.full_request IS NOT NULL",
            "T.full_response IS NOT NULL",
            "JSON_VALUE(T.metadata.request_latency) IS NOT NULL"
        ]
        
        if model_name:
            where_clauses.append(f"T.model LIKE '%{model_name}%'")
        
        where_clause = " AND ".join(where_clauses)
        
        query = f"""
        SELECT
          CAST(JSON_EXTRACT_SCALAR(T.metadata, '$.request_latency') AS FLOAT64) / 1000 AS latency,
          SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.promptTokenCount) AS INT64) AS input_tokens,
          SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.candidatesTokenCount) AS INT64) AS output_tokens,
          SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.thoughtsTokenCount) AS INT64) AS thought_tokens,
          SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.totalTokenCount) AS INT64) AS total_tokens
        FROM
          `{PROJECT_ID}.{DATASET}.{GEMINI_LOG_TABLE}` AS T
        WHERE
          {where_clause}
        LIMIT 5000
        """
        
        df = execute_bigquery(query)
        
        if df.empty or len(df) < 10:
            logging.warning(f"Insufficient data for correlation analysis: {len(df) if not df.empty else 0} rows")
            return json.dumps({"error": "Insufficient data for correlation analysis"})
        
        logging.info(f"Analyzing correlation for {len(df)} data points")
        
        # Calculate output + thought tokens
        df['output_thought_tokens'] = df['output_tokens'].fillna(0) + df['thought_tokens'].fillna(0)
        
        # Calculate all correlations
        correlations = {}
        
        for col in ['input_tokens', 'output_tokens', 'thought_tokens', 'output_thought_tokens', 'total_tokens']:
            if df[col].notna().sum() > 10:
                corr = df['latency'].corr(df[col])
                correlations[f"latency_vs_{col}"] = {
                    "correlation": float(corr) if pd.notna(corr) else None,
                    "strength": "strong" if abs(corr) > 0.7 else "moderate" if abs(corr) > 0.4 else "weak" if abs(corr) > 0.2 else "negligible",
                    "direction": "positive" if corr > 0 else "negative" if corr < 0 else "none"
                }
                logging.info(f"Correlation latency vs {col}: {corr:.3f} ({correlations[f'latency_vs_{col}']['strength']})")
        
        # Find strongest correlation
        strongest = max(correlations.items(), key=lambda x: abs(x[1]['correlation']) if x[1]['correlation'] is not None else 0)
        
        # Statistical breakdown by quartiles
        quartile_analysis = {}
        for col in ['output_thought_tokens', 'input_tokens']:
            if df[col].notna().sum() > 10:
                try:
                    # Try to create quartiles, but handle cases with insufficient unique values
                    df[f'{col}_quartile'] = pd.qcut(df[col], q=4, labels=['Q1_low', 'Q2_med_low', 'Q3_med_high', 'Q4_high'], duplicates='drop')
                    quartile_stats = df.groupby(f'{col}_quartile', observed=True)['latency'].agg(['mean', 'median', 'count']).to_dict('index')
                    quartile_analysis[col] = {k: {
                        "mean_latency": float(v['mean']),
                        "median_latency": float(v['median']),
                        "count": int(v['count'])
                    } for k, v in quartile_stats.items()}
                except (ValueError, TypeError) as e:
                    # If quartiles can't be created (e.g., too few unique values), skip this column
                    logging.warning(f"Could not create quartiles for {col}: {str(e)}")
                    quartile_analysis[col] = {"error": f"Insufficient unique values for quartile analysis: {str(e)}"}

        
        result = {
            "metadata": {
                "time_range": f"{start_time} to {end_time}",
                "sample_size": len(df)
            },
            "correlations": correlations,
            "strongest_correlation": {
                "metric": strongest[0],
                "value": strongest[1]['correlation'],
                "strength": strongest[1]['strength']
            },
            "quartile_analysis": quartile_analysis,
            "key_findings": [
                f"Latency vs output+thought tokens: {correlations.get('latency_vs_output_thought_tokens', {}).get('correlation', 0):.3f} ({correlations.get('latency_vs_output_thought_tokens', {}).get('strength', 'unknown')})",
                f"Latency vs input tokens: {correlations.get('latency_vs_input_tokens', {}).get('correlation', 0):.3f} ({correlations.get('latency_vs_input_tokens', {}).get('strength', 'unknown')})",
                f"Strongest predictor: {strongest[0].replace('latency_vs_', '')} (r={strongest[1]['correlation']:.3f})"
            ],
            "summary": f"Strongest correlation: {strongest[0]} (r={strongest[1]['correlation']:.3f}, {strongest[1]['strength']})"
        }
        
        return json.dumps(result, cls=AnalysisEncoder)
        
    except Exception as e:
        logging.error(f"Error in analyze_correlation_detailed: {str(e)}")
        return json.dumps({"error": str(e)})


def fetch_slow_queries(num_records: int = 10) -> str:
    """
    Fetches the top N slowest queries from the BigQuery logs and returns metadata only.
    
    This is a lightweight version that returns only request IDs and latency,
    avoiding token limits when dealing with large request/response payloads.
    
    Args:
        num_records: The number of records to fetch. Defaults to 10.
        
    Returns:
        A JSON string containing the count and list of request IDs with latency.
    """
    try:
        query = f"""
        SELECT
          CAST(T.request_id AS STRING) AS request_id,
          ROUND(SAFE_CAST(JSON_VALUE(T.metadata.request_latency) AS FLOAT64) / 1000.0, 2) AS request_latency_seconds
        FROM
          `{PROJECT_ID}.{DATASET}.{GEMINI_LOG_TABLE}` AS T
        WHERE
          T.full_request IS NOT NULL
          AND T.full_response IS NOT NULL
        ORDER BY
          request_latency_seconds DESC
        LIMIT {num_records}
        """
        
        df = execute_bigquery(query)
        
        if df.empty:
            return json.dumps({"error": "No data found"})
        
        # Convert results to a list of request IDs
        request_ids = []
        for _, row in df.iterrows():
            request_ids.append({
                "request_id": str(row["request_id"]),
                "latency_seconds": float(row["request_latency_seconds"]) if pd.notna(row["request_latency_seconds"]) else 0.0
            })
        
        logging.info(f"Successfully fetched {len(request_ids)} request IDs")
        return json.dumps({
            "count": len(request_ids),
            "requests": request_ids
        }, cls=AnalysisEncoder)
    
    except Exception as e:
        error_msg = f"Error fetching slow queries: {str(e)}"
        logging.error(error_msg)
        return json.dumps({"error": error_msg})


def fetch_single_query(request_id: str) -> str:
    """
    Fetches a single query's full details by request_id.
    
    This allows deep-dive analysis of individual slow queries with full
    request/response content. Use this after fetch_slow_queries to analyze
    specific queries one at a time to avoid token limits.
    
    Args:
        request_id: The request ID to fetch.
        
    Returns:
        A JSON string containing the full query details.
    """
    try:
        query = f"""
        SELECT
          T.logging_time,
          CAST(T.request_id AS STRING) AS request_id,
          T.full_request,
          T.full_response,
          T.model,
          JSON_VALUE(T.full_request.labels.adk_agent_name) AS adk_agent_name,
          ROUND(SAFE_CAST(JSON_VALUE(T.metadata.request_latency) AS FLOAT64) / 1000.0, 2) AS request_latency_seconds,
          SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.thoughtsTokenCount) AS INT64) AS thoughts_token_count,
          SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.candidatesTokenCount) AS INT64) AS output_token_count,
          SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.promptTokenCount) AS INT64) AS prompt_token_count,
          SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.totalTokenCount) AS INT64) AS total_token_count
        FROM
          `{PROJECT_ID}.{DATASET}.{GEMINI_LOG_TABLE}` AS T
        WHERE
          CAST(T.request_id AS STRING) = @request_id
        LIMIT 1
        """
        
        client = bigquery.Client(project=PROJECT_ID)
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("request_id", "STRING", str(request_id))
            ]
        )
        
        df = client.query(query, job_config=job_config).to_dataframe()
        
        if df.empty:
            return json.dumps({"error": f"No record found for request_id: {request_id}"})
        
        row = df.iloc[0]
        record = {
            "logging_time": row['logging_time'].isoformat() if pd.notna(row['logging_time']) else None,
            "request_id": row['request_id'],
            "full_request": json.loads(row['full_request']) if pd.notna(row['full_request']) else None,
            "full_response": json.loads(row['full_response']) if pd.notna(row['full_response']) else None,
            "model": row['model'],
            "adk_agent_name": row['adk_agent_name'] if pd.notna(row['adk_agent_name']) else None,
            "request_latency_seconds": float(row['request_latency_seconds']) if pd.notna(row['request_latency_seconds']) else None,
            "thoughts_token_count": int(row['thoughts_token_count']) if pd.notna(row['thoughts_token_count']) else None,
            "output_token_count": int(row['output_token_count']) if pd.notna(row['output_token_count']) else None,
            "prompt_token_count": int(row['prompt_token_count']) if pd.notna(row['prompt_token_count']) else None,
            "total_token_count": int(row['total_token_count']) if pd.notna(row['total_token_count']) else None
        }
        
        logging.info(f"Successfully fetched query {request_id}")
        return json.dumps(record, cls=AnalysisEncoder, default=str)
    
    except Exception as e:
        error_msg = f"Error fetching query {request_id}: {str(e)}"
        logging.error(error_msg)
        return json.dumps({"error": error_msg})


def save_analysis_report(
    report_content: str,
    filename: str = "latency_analysis_report.md"
) -> str:
    """
    Save the analysis report to a markdown file.
    
    This tool allows the agent to save its final comprehensive report to a file
    for easy sharing and documentation.
    
    Args:
        report_content: The markdown-formatted report content to save
        filename: The filename to save the report as (default: latency_analysis_report.md)
        
    Returns:
        A JSON string confirming the save or reporting an error
    """
    try:
        import os
        from datetime import datetime
        
        # Create reports directory if it doesn't exist
        reports_dir = os.path.join(os.path.dirname(__file__), "../../reports")
        os.makedirs(reports_dir, exist_ok=True)
        
        # Add timestamp to filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = filename.replace(".md", "")
        timestamped_filename = f"{base_name}_{timestamp}.md"
        
        filepath = os.path.join(reports_dir, timestamped_filename)
        
        # Write the report
        with open(filepath, 'w') as f:
            f.write(report_content)
        
        logging.info(f"Successfully saved report to {filepath}")
        
        return json.dumps({
            "success": True,
            "filepath": filepath,
            "filename": timestamped_filename,
            "message": f"Report successfully saved to {filepath}"
        })
    
    except Exception as e:
        error_msg = f"Error saving report: {str(e)}"
        logging.error(error_msg)
        return json.dumps({"error": error_msg})


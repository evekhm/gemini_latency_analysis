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
from google.api_core.exceptions import GoogleAPICallError
from .query_extractor import extract_user_query
from dotenv import load_dotenv

load_dotenv()

PROJECT_ID = os.getenv('PROJECT_ID')
DATASET = os.getenv('DATASET', 'gemini_logs')
GEMINI_LOG_TABLE = os.getenv('GEMINI_LOG_TABLE', 'gemini_logs')

# Agent version for tracking
AGENT_VERSION = "1.1.0"  # Added KPI compliance and request queuing analysis

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


def parse_time_range(time_range: str) -> str:
    """
    Parse time range string into start and end timestamps.
    
    Supports formats:
    - "24h", "7d" (last N hours/days)
    - "24h ago", "7d ago" (relative past)
    - "YYYY-MM-DD" (specific date)
    - "now" (current time)
    - "2 september" (natural language)
    - Ranges: "start to end", "from start to end"
    """
    from dateutil import parser
    from dateutil.relativedelta import relativedelta
    
    now = datetime.utcnow()
    time_range = time_range.strip().lower()
    
    if time_range == 'all':
        start = datetime(2000, 1, 1)
        end = now
        return json.dumps({"start_date": start.strftime('%Y-%m-%d %H:%M:%S'), "end_date": end.strftime('%Y-%m-%d %H:%M:%S')})

    
    # Strip "from " prefix if present
    if time_range.startswith('from '):
        time_range = time_range[5:].strip()
    
    # Helper to parse single date point
    def parse_point(s: str) -> datetime:
        s = s.strip()
        if s == 'now':
            return now
        
        # Handle relative "ago" formats
        if s.endswith(' ago'):
            s = s[:-4].strip()
        
        # Handle simple relative formats (with or without "ago")
        if s.endswith('h'):
            try:
                return now - timedelta(hours=int(s[:-1]))
            except ValueError:
                pass
        if s.endswith('d'):
            try:
                return now - timedelta(days=int(s[:-1]))
            except ValueError:
                pass
        
        # Handle "last X days/hours/months"
        if s.startswith('last '):
            val = s[5:].strip()
            if val.endswith(' days'):
                try:
                    return now - timedelta(days=int(val[:-5]))
                except ValueError:
                    pass
            if val.endswith(' hours'):
                try:
                    return now - timedelta(hours=int(val[:-6]))
                except ValueError:
                    pass
            if val.endswith(' month'):
                 return now - relativedelta(months=1)
            if val.endswith(' months'):
                try:
                    return now - relativedelta(months=int(val[:-7]))
                except ValueError:
                    pass

        # Use dateutil for everything else (absolute dates, natural language)
        try:
            # default to current year if missing, fuzzy=True allows ignoring noise
            return parser.parse(s, default=now, fuzzy=True)
        except (ValueError, TypeError):
            pass
            
        # Fallback/Error
        raise ValueError(f"Could not parse date format: '{s}'")

    try:
        if ' to ' in time_range:
            parts = time_range.split(' to ')
            start_str, end_str = parts[0], parts[1]
        elif '-' in time_range and len(time_range.split('-')) == 3 and time_range.count('.') == 2: # Heuristic for DD.MM.YYYY-DD.MM.YYYY
             parts = time_range.split('-')
             if len(parts) == 2: # e.g. 10.10.2025-12.12.2025
                 start_str, end_str = parts[0], parts[1]
             else: # Likely not a range of this type
                 start_str, end_str = time_range, None
        else:
            start_str, end_str = time_range, None

        if end_str:
            start = parse_point(start_str)
            end = parse_point(end_str)
            # If end seems to be just a date, extend to end of day
            if end.hour == 0 and end.minute == 0 and end.second == 0 and len(end_str.strip()) <= 10:
                end += timedelta(days=1, microseconds=-1)
        else:
            # Single value: "24h" means last 24 hours (end=now)
            # "last month" means last month to now
            # "2 september" means 2 september to now
            start = parse_point(time_range)
            end = now
            # Adjust start if it's a duration like "24h" to be relative to end
            if time_range.endswith('h') or time_range.endswith('d') or time_range.startswith('last '):
                 # Re-parse with now=end to get the start relative to now
                 # This is a bit redundant, the parse_point already does this.
                 pass
            
    except Exception as e:
        logging.warning(f"Error parsing time range '{time_range}': {e}. Defaulting to 24h.")
        start = now - timedelta(hours=24)
        end = now
    
    return json.dumps({"start_date": start.strftime('%Y-%m-%d %H:%M:%S'), "end_date": end.strftime('%Y-%m-%d %H:%M:%S')})


def get_analysis_metadata() -> str:
    """
    Get metadata about the analysis environment for report headers.
    
    Returns actual values from environment variables to avoid hallucination.
    Use this tool when generating report metadata headers.
    
    Returns:
        JSON string with project_id, dataset, table, and analyzer version
    """
    from datetime import datetime
    
    metadata = {
        "project_id": PROJECT_ID,
        "dataset": DATASET,
        "table": GEMINI_LOG_TABLE,
        "analyzer_version": AGENT_VERSION,
        "generated_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    return json.dumps(metadata, cls=AnalysisEncoder)


def verify_data_access() -> str:
    """
    Verifies BigQuery configuration and data access.
    
    Use this tool when you encounter "No data found" errors to check:
    1. If the configuration (Project, Dataset, Table) is correct
    2. If the agent has permissions to access the table
    3. If the table actually contains data
    
    Returns:
        JSON string with configuration details and test query result
    """
    config = {
        "project_id": PROJECT_ID,
        "dataset": DATASET,
        "table": GEMINI_LOG_TABLE,
        "env_vars_loaded": {
            "PROJECT_ID": bool(PROJECT_ID),
            "DATASET": bool(DATASET),
            "GEMINI_LOG_TABLE": bool(GEMINI_LOG_TABLE)
        }
    }
    
    # Log configuration for visibility
    logging.info(f"[CONFIG] Verifying access with: Project={PROJECT_ID}, Dataset={DATASET}, Table={GEMINI_LOG_TABLE}")
    
    try:
        # Test query to check access and get total count
        query = f"""
        SELECT 
            COUNT(*) as total_rows,
            MIN(logging_time) as first_log,
            MAX(logging_time) as last_log
        FROM `{PROJECT_ID}.{DATASET}.{GEMINI_LOG_TABLE}`
        """
        
        df = execute_bigquery(query, timeout=30)
        
        if not df.empty:
            row = df.iloc[0]
            result = {
                "status": "SUCCESS",
                "message": "Successfully connected to BigQuery table",
                "total_rows": int(row['total_rows']),
                "data_range": {
                    "start": row['first_log'].isoformat() if pd.notna(row['first_log']) else None,
                    "end": row['last_log'].isoformat() if pd.notna(row['last_log']) else None
                },
                "configuration": config
            }
        else:
            result = {
                "status": "WARNING",
                "message": "Query executed but returned no rows",
                "configuration": config
            }
            
    except Exception as e:
        result = {
            "status": "ERROR",
            "message": f"Failed to access BigQuery: {str(e)}",
            "error_type": type(e).__name__,
            "configuration": config
        }
        logging.error(f"Data access verification failed: {str(e)}")
        
    return json.dumps(result, cls=AnalysisEncoder, default=str)


def execute_bigquery(query: str, timeout: int = 1200):
    """Execute BigQuery query with timeout protection.
    
    Args:
        query: SQL query to execute
        timeout: Timeout in seconds (default: 300s = 5 minutes)
        
    Returns:
        DataFrame with query results
        
    Raises:
        TimeoutError: If query exceeds timeout
    """
    if not PROJECT_ID:
        raise ValueError("PROJECT_ID environment variable is not set")
    
    client = bigquery.Client(project=PROJECT_ID)
    job_config = bigquery.QueryJobConfig(
        job_timeout_ms=timeout * 1000  # Convert to milliseconds
    )
    query_job = client.query(query, job_config=job_config)
    
    try:
        df = query_job.result(timeout=timeout).to_dataframe()
        return df
    except Exception as e:
        if "timeout" in str(e).lower():
            logging.error(f"BigQuery query timed out after {timeout} seconds")
        raise


def get_overall_statistics(
    time_range: str = "24h",
    model_name: Optional[str] = None,
    agent_name: Optional[str] = None
) -> str:
    """
    Get overall latency and token statistics with comprehensive percentile breakdown.
    
    Returns mean, median, P75, P90, P95, P99, P99.9 latency percentiles for
    complete distribution analysis.
    
    Args:
        time_range: Time range to analyze (e.g., "24h", "7d", "2025-01-01 to 2025-01-31")
        model_name: Filter by specific model (optional)
        agent_name: Filter by specific agent (optional)
        
    Returns:
        JSON string with overall statistics including all percentiles
    """
    try:
        time_range_dict = json.loads(parse_time_range(time_range))
        start_time, end_time = time_range_dict['start_date'], time_range_dict['end_date']
        
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
          APPROX_QUANTILES(CAST(JSON_EXTRACT_SCALAR(T.metadata, '$.request_latency') AS FLOAT64) / 1000, 100)[OFFSET(75)] AS p75_latency,
          APPROX_QUANTILES(CAST(JSON_EXTRACT_SCALAR(T.metadata, '$.request_latency') AS FLOAT64) / 1000, 100)[OFFSET(90)] AS p90_latency,
          APPROX_QUANTILES(CAST(JSON_EXTRACT_SCALAR(T.metadata, '$.request_latency') AS FLOAT64) / 1000, 100)[OFFSET(95)] AS p95_latency,
          APPROX_QUANTILES(CAST(JSON_EXTRACT_SCALAR(T.metadata, '$.request_latency') AS FLOAT64) / 1000, 100)[OFFSET(99)] AS p99_latency,
          APPROX_QUANTILES(CAST(JSON_EXTRACT_SCALAR(T.metadata, '$.request_latency') AS FLOAT64) / 1000, 1000)[OFFSET(999)] AS p999_latency,
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
                "p75": float(row['p75_latency']) if pd.notna(row['p75_latency']) else None,
                "p90": float(row['p90_latency']) if pd.notna(row['p90_latency']) else None,
                "p95": float(row['p95_latency']) if pd.notna(row['p95_latency']) else None,
                "p99": float(row['p99_latency']) if pd.notna(row['p99_latency']) else None,
                "p99.9": float(row['p999_latency']) if pd.notna(row['p999_latency']) else None,
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
        time_range_dict = json.loads(parse_time_range(time_range))
        start_time, end_time = time_range_dict['start_date'], time_range_dict['end_date']
        
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
        time_range_dict = json.loads(parse_time_range(time_range))
        start_time, end_time = time_range_dict['start_date'], time_range_dict['end_date']
        
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
        time_range_dict = json.loads(parse_time_range(time_range))
        start_time, end_time = time_range_dict['start_date'], time_range_dict['end_date']
        
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
        time_range_dict = json.loads(parse_time_range(time_range))
        start_time, end_time = time_range_dict['start_date'], time_range_dict['end_date']
        
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
    agent_name: Optional[str] = None,
    threshold_std: float = 3.0
) -> str:
    """
    Identify outlier requests (latency > threshold * std dev).
    
    Returns list of outlier requests with full details.
    """
    try:
        time_range_dict = json.loads(parse_time_range(time_range))
        start_time, end_time = time_range_dict['start_date'], time_range_dict['end_date']
        
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
    num_queries: int = 20,
    time_range: str = "24h",
    model_name: Optional[str] = None,
    agent_name: Optional[str] = None
) -> str:
    """
    Get the N slowest queries with full details for deep analysis.
    
    This is the merged functionality from slow_query_analyzer.
    Returns request IDs and metadata for the slowest queries.
    """
    try:
        time_range_dict = json.loads(parse_time_range(time_range))
        start_time, end_time = time_range_dict['start_date'], time_range_dict['end_date']
        
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
          CAST(T.request_id AS STRING) AS request_id,
          T.logging_time,
          CAST(JSON_EXTRACT_SCALAR(T.metadata, '$.request_latency') AS FLOAT64) / 1000 AS latency,
          T.model,
          COALESCE(JSON_VALUE(T.full_request.labels.adk_agent_name), 'unknown') AS agent_name,
          SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.promptTokenCount) AS INT64) AS input_tokens,
          SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.candidatesTokenCount) AS INT64) AS output_tokens,
          SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.thoughtsTokenCount) AS INT64) AS thought_tokens,
          SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.totalTokenCount) AS INT64) AS total_tokens,
          -- Extract last user message text (avoid fetching entire JSON)
          (
            SELECT STRING_AGG(JSON_VALUE(part, '$.text'), ' ')
            FROM UNNEST(JSON_QUERY_ARRAY(T.full_request, '$.contents')) AS content,
                 UNNEST(JSON_QUERY_ARRAY(content, '$.parts')) AS part
            WHERE JSON_VALUE(content, '$.role') = 'user'
            ORDER BY OFFSET(content) DESC
            LIMIT 1
          ) AS last_user_message
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
            # Extract query preview from last user message
            query_preview = "N/A"
            if pd.notna(row['last_user_message']):
                msg = row['last_user_message']
                # Apply context stripping logic
                if "</Context>" in msg:
                    msg = msg.split("</Context>")[-1].strip()
                elif "</context>" in msg:
                    msg = msg.split("</context>")[-1].strip()
                
                if len(msg) > 500 and "for the question" in msg.lower():
                    idx = msg.lower().find("for the question")
                    msg = msg[idx:].strip()
                
                query_preview = msg[:150]

            queries.append({
                "request_id": row['request_id'],
                "timestamp": row['logging_time'].isoformat() if pd.notna(row['logging_time']) else None,
                "latency": float(row['latency']),
                "model": row['model'],
                "agent_name": row['agent_name'],
                "input_tokens": int(row['input_tokens']) if pd.notna(row['input_tokens']) else None,
                "output_tokens": int(row['output_tokens']) if pd.notna(row['output_tokens']) else None,
                "thought_tokens": int(row['thought_tokens']) if pd.notna(row['thought_tokens']) else None,
                "total_tokens": int(row['total_tokens']) if pd.notna(row['total_tokens']) else None,
                "query_preview": query_preview
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
        time_range_dict = json.loads(parse_time_range(time_range))
        start_time, end_time = time_range_dict['start_date'], time_range_dict['end_date']
        
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
        time_range_dict = json.loads(parse_time_range(time_range))
        start_time, end_time = time_range_dict['start_date'], time_range_dict['end_date']
        
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
    model_name: Optional[str] = None,
    agent_name: Optional[str] = None
) -> str:
    """
    Analyze token usage and estimated costs.
    
    Provides cost breakdown by agent and identifies expensive operations.
    """
    try:
        time_range_dict = json.loads(parse_time_range(time_range))
        start_time, end_time = time_range_dict['start_date'], time_range_dict['end_date']
        
        where_clauses = [
            f"T.logging_time BETWEEN '{start_time}' AND '{end_time}'",
            "T.full_request IS NOT NULL",
            "T.full_response IS NOT NULL"
        ]
        
        if model_name:
            where_clauses.append(f"T.model LIKE '%{model_name}%'")
        if agent_name:
            where_clauses.append(f"JSON_VALUE(T.full_request.labels.adk_agent_name) = '{agent_name}'")
        
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
    model_name: Optional[str] = None,
    agent_name: Optional[str] = None
) -> str:
    """
    Compare performance between two time periods.
    
    Useful for before/after analysis or A/B testing validation.
    """
    try:
        time_range_dict1 = json.loads(parse_time_range(period1))
        start1, end1 = time_range_dict1['start_date'], time_range_dict1['end_date']

        time_range_dict2 = json.loads(parse_time_range(period2))
        start2, end2 = time_range_dict2['start_date'], time_range_dict2['end_date']
        
        def get_period_stats(start, end):
            where_clauses = [
                f"T.logging_time BETWEEN '{start}' AND '{end}'",
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
    model_name: Optional[str] = None,
    agent_name: Optional[str] = None
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
        time_range_dict = json.loads(parse_time_range(time_range))
        start_time, end_time = time_range_dict['start_date'], time_range_dict['end_date']
        
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
    model_name: Optional[str] = None,
    agent_name: Optional[str] = None
) -> str:
    """
    Detailed correlation analysis including output+thought tokens.
    
    Provides comprehensive correlation matrix and statistical significance.
    """
    try:
        time_range_dict = json.loads(parse_time_range(time_range))
        start_time, end_time = time_range_dict['start_date'], time_range_dict['end_date']
        
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


def fetch_slow_queries(num_records: int = 20, agent_name: Optional[str] = None) -> str:
    """
    Fetches the top N slowest queries from the BigQuery logs and returns metadata only.
    
    This is a lightweight version that returns only request IDs and latency,
    avoiding token limits when dealing with large request/response payloads.
    
    Args:
        num_records: The number of records to fetch. Defaults to 10.
        agent_name: Filter by specific agent (optional)
        
    Returns:
        A JSON string containing the count and list of request IDs with latency.
    """
    logging.info(f"[PROGRESS] Starting fetch of {num_records} slowest queries")
    try:
        where_clause = "T.full_request IS NOT NULL AND T.full_response IS NOT NULL"
        if agent_name:
            where_clause += f" AND JSON_VALUE(T.full_request.labels.adk_agent_name) = '{agent_name}'"

        query = f"""
        SELECT
          CAST(T.request_id AS STRING) AS request_id,
          ROUND(SAFE_CAST(JSON_VALUE(T.metadata.request_latency) AS FLOAT64) / 1000.0, 2) AS request_latency_seconds
        FROM
          `{PROJECT_ID}.{DATASET}.{GEMINI_LOG_TABLE}` AS T
        WHERE
          {where_clause}
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
        
        logging.info(f"[PROGRESS] Successfully fetched {len(request_ids)} request IDs")
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
    logging.info(f"[PROGRESS] Starting fetch for query {request_id}")
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
        
        logging.info(f"[PROGRESS] Successfully fetched query {request_id}")
        return json.dumps(record, cls=AnalysisEncoder, default=str)
    
    except Exception as e:
        error_msg = f"Error fetching query {request_id}: {str(e)}"
        logging.error(f"[PROGRESS] Failed to fetch query {request_id}: {str(e)}")
        return json.dumps({"error": error_msg})


def fetch_slow_queries_batch(
    num_queries: int = 20,
    time_range: str = "24h",
    model_name: Optional[str] = None,
    agent_name: Optional[str] = None
) -> str:
    """
    Fetches multiple slow queries with full details in a SINGLE batch query.
    
    This is the RECOMMENDED approach instead of calling fetch_single_query() 
    multiple times, as it:
    - Avoids sequential LLM calls that can timeout
    - Fetches all data in one BigQuery query
    - Returns complete information for analysis
    
    Use this function when you need to analyze multiple slow queries.
    Only use fetch_single_query() for 1-2 specific examples.
    
    Args:
        num_queries: Number of slowest queries to fetch (default: 20)
        time_range: Time range to search (default: "24h")
        model_name: Optional model name filter
        agent_name: Optional agent name filter
        
    Returns:
        JSON string with array of query details including full request/response
    """
    # Cap number of queries to prevent massive payloads that cause timeouts
    MAX_QUERIES = 20
    if num_queries > MAX_QUERIES:
        logging.warning(f"[CONFIG] Capping batch size from {num_queries} to {MAX_QUERIES} to prevent timeouts")
        num_queries = MAX_QUERIES
        
    logging.info(f"[PROGRESS] Starting batch fetch of {num_queries} slowest queries")
    try:
        time_range_dict = json.loads(parse_time_range(time_range))
        start_time, end_time = time_range_dict['start_date'], time_range_dict['end_date']
        
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
          SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.totalTokenCount) AS INT64) AS total_token_count,
          SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.totalTokenCount) AS INT64) AS total_token_count,
          -- Extract last 500 chars of prompt to capture user question (generic approach)
          SUBSTR(JSON_VALUE(T.full_request.contents[0].parts[0].text), -500) AS query_preview
        FROM
          `{PROJECT_ID}.{DATASET}.{GEMINI_LOG_TABLE}` AS T
        WHERE
          {where_clause}
        ORDER BY request_latency_seconds DESC
        LIMIT {num_queries}
        """
        
        df = execute_bigquery(query)
        
        if df.empty:
            return json.dumps({"error": "No slow queries found"})
        
        # Convert to list of records
        queries = []
        for _, row in df.iterrows():
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
                "total_token_count": int(row['total_token_count']) if pd.notna(row['total_token_count']) else None,
                "query_preview": row['query_preview'] if pd.notna(row['query_preview']) else None
            }
            queries.append(record)
        
        logging.info(f"[PROGRESS] Successfully fetched {len(queries)} slow queries in batch")
        
        result = {
            "count": len(queries),
            "queries": queries,
            "metadata": {
                "analyzed_count": len(queries),
                "requested_count": num_queries if num_queries <= MAX_QUERIES else 50, # Approximate original request if capped
                "time_range": f"{start_time} to {end_time}",
                "model_filter": model_name if model_name else "all models"
            }
        }
        
        if num_queries == MAX_QUERIES:
             result["metadata"]["warning"] = f"Batch size limited to {MAX_QUERIES} to prevent LLM timeouts. Displaying top {MAX_QUERIES} slowest queries out of requested batch."
             
        return json.dumps(result, cls=AnalysisEncoder, default=str)
    
    except Exception as e:
        error_msg = f"Error in batch fetch: {str(e)}"
        logging.error(f"[PROGRESS] Failed to fetch slow queries batch: {str(e)}")
        return json.dumps({"error": error_msg})


def fetch_fastest_queries(
    num_queries: int = 20,
    time_range: str = "24h",
    model_name: Optional[str] = None,
    agent_name: Optional[str] = None
) -> str:
    """
    Fetches multiple FASTEST queries (low latency) to serve as a baseline.
    
    Use this to compare against slow queries. If a factor (like input tokens) 
    is high in BOTH slow and fast queries, it is NOT the driver of latency.
    
    Args:
        num_queries: Number of fastest queries to fetch (default: 20)
        time_range: Time range to search (default: "24h")
        model_name: Optional model name filter
        
    Returns:
        JSON string with array of query details including full request/response
    """
    # Cap number of queries to prevent massive payloads
    MAX_QUERIES = 20
    if num_queries > MAX_QUERIES:
        logging.warning(f"[CONFIG] Capping batch size from {num_queries} to {MAX_QUERIES} to prevent timeouts")
        num_queries = MAX_QUERIES
        
    logging.info(f"[PROGRESS] Starting batch fetch of {num_queries} FASTEST queries (baseline)")
    try:
        time_range_dict = json.loads(parse_time_range(time_range))
        start_time, end_time = time_range_dict['start_date'], time_range_dict['end_date']
        
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
          SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.totalTokenCount) AS INT64) AS total_token_count,
          -- Extract last 500 chars of prompt to capture user question (generic approach)
          SUBSTR(JSON_VALUE(T.full_request.contents[0].parts[0].text), -500) AS query_preview
        FROM
          `{PROJECT_ID}.{DATASET}.{GEMINI_LOG_TABLE}` AS T
        WHERE
          {where_clause}
        ORDER BY request_latency_seconds ASC
        LIMIT {num_queries}
        """
        
        df = execute_bigquery(query)
        
        if df.empty:
            return json.dumps({"error": "No queries found"})
        
        # Convert to list of records
        queries = []
        for _, row in df.iterrows():
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
                "total_token_count": int(row['total_token_count']) if pd.notna(row['total_token_count']) else None,
                "query_preview": row['query_preview'] if pd.notna(row['query_preview']) else None
            }
            queries.append(record)
        
        logging.info(f"[PROGRESS] Successfully fetched {len(queries)} FASTEST queries in batch")
        
        result = {
            "count": len(queries),
            "queries": queries,
            "metadata": {
                "analyzed_count": len(queries),
                "requested_count": num_queries if num_queries <= MAX_QUERIES else 50,
                "time_range": f"{start_time} to {end_time}",
                "model_filter": model_name if model_name else "all models",
                "type": "fastest_queries_baseline"
            }
        }
        
        if num_queries == MAX_QUERIES:
             result["metadata"]["warning"] = f"Batch size limited to {MAX_QUERIES} to prevent LLM timeouts."
             
        return json.dumps(result, cls=AnalysisEncoder, default=str)
    
    except Exception as e:
        error_msg = f"Error in batch fetch: {str(e)}"
        logging.error(f"[PROGRESS] Failed to fetch fastest queries batch: {str(e)}")
        return json.dumps({"error": error_msg})


def get_token_velocity(
    time_range: str = "7d",
    model_name: Optional[str] = None,
    agent_name: Optional[str] = None
) -> str:
    """
    Analyze Time Per Output Token (TPOT) to distinguish between slow generation and verbose output.
    
    TPOT = Latency / Output Tokens
    
    High TPOT (> 0.1s/token) indicates compute bottlenecks (model struggling).
    Low TPOT (< 0.05s/token) but high latency indicates verbose output (generating too much text).
    """
    try:
        time_range_dict = json.loads(parse_time_range(time_range))
        start_time, end_time = time_range_dict['start_date'], time_range_dict['end_date']
        
        where_clauses = [
            f"T.logging_time BETWEEN '{start_time}' AND '{end_time}'",
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
          CAST(JSON_EXTRACT_SCALAR(T.metadata, '$.request_latency') AS FLOAT64) / 1000 AS latency,
          SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.candidatesTokenCount) AS INT64) AS output_tokens
        FROM
          `{PROJECT_ID}.{DATASET}.{GEMINI_LOG_TABLE}` AS T
        WHERE
          {where_clause}
          AND SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.candidatesTokenCount) AS INT64) > 0
        LIMIT 5000
        """
        
        df = execute_bigquery(query)
        
        if df.empty:
            return json.dumps({"error": "No data found for TPOT analysis"})
            
        # Calculate TPOT (seconds per token)
        df['tpot'] = df['latency'] / df['output_tokens']

        # Statistics
        stats = {
            "avg_tpot": float(df['tpot'].mean()),
            "median_tpot": float(df['tpot'].median()),
            "p95_tpot": float(df['tpot'].quantile(0.95)),
            "min_tpot": float(df['tpot'].min()),
            "max_tpot": float(df['tpot'].max())
        }
        
        # Categorize requests
        # Fast generation: < 50ms/token
        # Normal generation: 50-100ms/token
        # Slow generation: > 100ms/token
        df['speed_category'] = pd.cut(
            df['tpot'], 
            bins=[0, 0.05, 0.1, float('inf')], 
            labels=['fast_compute', 'normal_compute', 'slow_compute']
        )
        
        speed_breakdown = df['speed_category'].value_counts(normalize=True).to_dict()
        
        # Correlation: Does TPOT correlate with total latency?
        # If yes, the model is struggling. If no, it's just token volume.
        tpot_latency_corr = float(df['tpot'].corr(df['latency']))
        
        result = {
            "metadata": {
                "time_range": f"{start_time} to {end_time}",
                "data_points": len(df)
            },
            "statistics": stats,
            "speed_breakdown": {k: float(v) for k, v in speed_breakdown.items()},
            "correlations": {
                "tpot_vs_latency": tpot_latency_corr,
                "interpretation": "High correlation means slow compute drives latency. Low correlation means token volume drives latency."
            },
            "insights": []
        }
        
        # Generate insights
        if stats['avg_tpot'] > 0.1:
            result['insights'].append("High average TPOT (>100ms/token) indicates potential compute bottlenecks or model overload.")
        elif stats['avg_tpot'] < 0.05:
            result['insights'].append("Low average TPOT (<50ms/token) indicates efficient generation. High latency is likely due to verbose output.")
            
        if speed_breakdown.get('slow_compute', 0) > 0.2:
            result['insights'].append(f"Significant portion ({speed_breakdown['slow_compute']:.1%}) of requests suffer from slow generation speeds.")
            
        return json.dumps(result, cls=AnalysisEncoder)
        
    except Exception as e:
        logging.error(f"Error in get_token_velocity: {str(e)}")
        return json.dumps({"error": str(e)})


def analyze_request_queuing(
    time_range: str = "24h",
    model_name: Optional[str] = None,
    agent_name: Optional[str] = None,
    burst_window_seconds: int = 1
) -> str:
    """
    Analyze if latency spikes are caused by request queuing (micro-bursts).
    
    Checks if multiple requests arriving within a small window (e.g. 1s)
    lead to increased latency, suggesting a queuing mechanism is delaying execution.
    """
    try:
        time_range_dict = json.loads(parse_time_range(time_range))
        start_time, end_time = time_range_dict['start_date'], time_range_dict['end_date']
        
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
        
        # Group by small time windows (micro-bursts)
        query = f"""
        WITH bursts AS (
          SELECT
            TIMESTAMP_TRUNC(T.logging_time, SECOND) AS burst_time,
            COUNT(*) as request_count,
            AVG(CAST(JSON_EXTRACT_SCALAR(T.metadata, '$.request_latency') AS FLOAT64) / 1000) AS avg_latency,
            MAX(CAST(JSON_EXTRACT_SCALAR(T.metadata, '$.request_latency') AS FLOAT64) / 1000) AS max_latency
          FROM
            `{PROJECT_ID}.{DATASET}.{GEMINI_LOG_TABLE}` AS T
          WHERE
            {where_clause}
          GROUP BY burst_time
        )
        SELECT
          request_count,
          COUNT(*) as burst_count,
          AVG(avg_latency) as mean_latency_for_burst_size,
          AVG(max_latency) as mean_max_latency_for_burst_size
        FROM bursts
        GROUP BY request_count
        ORDER BY request_count
        """
        
        df = execute_bigquery(query)
        
        if df.empty:
            return json.dumps({"error": "No data found for queuing analysis"})
            
        # Analyze correlation between burst size and latency
        correlation = df['request_count'].corr(df['mean_latency_for_burst_size'])
        
        burst_impact = []
        for _, row in df.iterrows():
            burst_impact.append({
                "concurrent_requests_per_sec": int(row['request_count']),
                "frequency": int(row['burst_count']),
                "avg_latency": float(row['mean_latency_for_burst_size']),
                "avg_max_latency": float(row['mean_max_latency_for_burst_size'])
            })
            
        # Determine if queuing is happening
        # If latency increases significantly with burst size, queuing is likely
        is_queuing = correlation > 0.7 if pd.notna(correlation) else False
        
        result = {
            "metadata": {
                "time_range": f"{start_time} to {end_time}",
                "burst_window_seconds": burst_window_seconds
            },
            "correlation": float(correlation) if pd.notna(correlation) else None,
            "queuing_detected": bool(is_queuing),
            "burst_impact": burst_impact,
            "summary": f"Queuing hypothesis {'SUPPORTED' if is_queuing else 'REJECTED'}. Correlation between burst size and latency: {correlation:.3f}"
        }
        
        return json.dumps(result, cls=AnalysisEncoder)
        
    except Exception as e:
        logging.error(f"Error in analyze_request_queuing: {str(e)}")
        return json.dumps({"error": str(e)})


def check_kpi_compliance(
    time_range: Optional[str] = None,
    mean_latency_target: Optional[float] = None,
    p95_latency_target: Optional[float] = None,
    agent_name: Optional[str] = None
) -> str:
    """
    Check if current performance meets defined KPIs.
    
    Args:
        time_range: Time range to analyze (defaults to config or "24h")
        mean_latency_target: Target for mean latency (defaults to config or 3.0)
        p95_latency_target: Target for P95 latency (defaults to config or 5.0)
        agent_name: Filter by specific agent (defaults to config or None)
    """
    try:
        # Load config to get defaults
        config_str = get_analysis_config()
        config = json.loads(config_str) if not config_str.startswith("Error") else {}
        
        # Apply defaults from config if not provided
        if time_range is None:
            days = config.get("time_period", 1)
            time_range = f"{days}d"
            
        if mean_latency_target is None:
            mean_latency_target = config.get("kpis", {}).get("mean_latency_target", 3.0)
            
        if p95_latency_target is None:
            p95_latency_target = config.get("kpis", {}).get("p95_latency_target", 5.0)
            
        if agent_name is None:
            agent_name = config.get("agent_name")

        # Reuse get_overall_statistics to get current metrics
        stats_json = get_overall_statistics(time_range=time_range, agent_name=agent_name)
        stats = json.loads(stats_json)
        
        if "error" in stats:
            return stats_json
            
        current_mean = stats['latency']['mean']
        current_p95 = stats['latency']['p95']
        
        compliance = {
            "mean_latency": {
                "target": mean_latency_target,
                "actual": current_mean,
                "status": "PASS" if current_mean <= mean_latency_target else "FAIL",
                "gap": current_mean - mean_latency_target
            },
            "p95_latency": {
                "target": p95_latency_target,
                "actual": current_p95,
                "status": "PASS" if current_p95 <= p95_latency_target else "FAIL",
                "gap": current_p95 - p95_latency_target
            },
            "overall_status": "PASS" if (current_mean <= mean_latency_target and current_p95 <= p95_latency_target) else "FAIL"
        }
        
        result = {
            "metadata": {
                "time_range": time_range,
                "agent_name": agent_name
            },
            "compliance": compliance,
            "summary": f"KPI Status: {compliance['overall_status']}. Mean: {current_mean:.2f}s (Target {mean_latency_target}s), P95: {current_p95:.2f}s (Target {p95_latency_target}s)"
        }
        
        return json.dumps(result, cls=AnalysisEncoder)
        
    except Exception as e:
        logging.error(f"Error in check_kpi_compliance: {str(e)}")
        return json.dumps({"error": str(e)})


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
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = filename.replace(".md", "")
        timestamped_filename = f"{base_name}_{timestamp}.md"
        
        filepath = os.path.join(reports_dir, timestamped_filename)
        
        # Write the report
        with open(filepath, 'w') as f:
            f.write(report_content)
        
        # Log with prominent formatting
        logging.info("=" * 80)
        logging.info("✅ REPORT GENERATED SUCCESSFULLY")
        logging.info("=" * 80)
        logging.info(f"Report saved to: {filepath}")
        logging.info(f"Filename: {timestamped_filename}")
        logging.info(f"Report size: {len(report_content)} characters")
        logging.info("=" * 80)
        
        return json.dumps({
            "success": True,
            "filepath": filepath,
            "filename": timestamped_filename,
            "message": f"Report successfully saved to {filepath}"
        })
    
    except Exception as e:
        error_msg = f"Error saving report: {str(e)}"
        
        # Log error with prominent formatting
        logging.error("=" * 80)
        logging.error("❌ REPORT GENERATION FAILED")
        logging.error("=" * 80)
        logging.error(f"Error: {error_msg}")
        logging.error(f"Attempted filename: {filename}")
        logging.error("=" * 80)
        
        return json.dumps({"error": error_msg})


def get_analysis_config() -> str:
    """Reads the analysis configuration from config files (.json).
    
    This tool reads configuration parameters like time_period, KPIs, agent_name,
    num_slowest_queries, and analysis_scope from the config file.
    
    NOTE: As of the refactoring, workflow logic has been moved to the system prompt.
    The agent should read the config and use analysis_scope to determine which workflow
    to follow (standard, autonomous, or deep_research).
    
    Returns:
        JSON string containing the configuration (time period, KPIs, agent filter, analysis_scope, etc.).
    """
    try:
        # Assuming the config file is in the parent directory of agents/latency_analyzer
        # agents/latency_analyzer/utils.py -> ../../autonomous_analysis_90d.json
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "autonomous_analysis_90d.json")
        
        if not os.path.exists(config_path):
             # Fallback to current directory or relative path if running from root
             config_path = "autonomous_analysis_90d.json"
             
        if not os.path.exists(config_path):
            error_msg = f"Config file not found at {config_path}"
            logging.error(error_msg)
            return json.dumps({"error": error_msg})

        with open(config_path, 'r') as f:
            data = json.load(f)
            # Return only the config section if it exists, otherwise the whole file
            config = data.get("config", data)
            
            # Log configuration settings for visibility
            logging.info("=" * 80)
            logging.info("ANALYSIS CONFIGURATION")
            logging.info("=" * 80)
            logging.info(f"Config file: {config_path}")
            logging.info(f"Time Period: {config.get('time_period_days', 'NOT SET')}")
            logging.info(f"Analysis Scope: {config.get('analysis_scope', 'NOT SET')}")
            logging.info(f"Target Mean Latency: {config.get('kpis', {}).get('mean_latency_target', 'NOT SET')}s")
            logging.info(f"Target P95 Latency: {config.get('kpis', {}).get('p95_latency_target', 'NOT SET')}s")
            logging.info(f"Num Slowest Queries: {config.get('num_slowest_queries', 'NOT SET')}")
            logging.info(f"Agent Filter: {config.get('agent_name', 'None (all agents)')}")
            logging.info("=" * 80)
            
            return json.dumps(config)
    except Exception as e:
        error_msg = f"Error reading config: {str(e)}"
        logging.error(error_msg)
        return json.dumps({"error": error_msg})

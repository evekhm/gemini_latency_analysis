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
from google.adk.tools.tool_context import ToolContext
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse

load_dotenv()

PROJECT_ID = os.getenv('PROJECT_ID')
DATASET_ID = os.getenv('DATASET_ID')
TABLE_ID = os.getenv('AGENT_TABLE_ID') or 'gemini_logs'

assert PROJECT_ID, "PROJECT_ID environment variable not set"
assert DATASET_ID, "DATASET_ID environment variable not set"
# TABLE_ID now defaults to 'gemini_logs' -> assertion removed

# Agent version for tracking
AGENT_VERSION = "0.0.1"

print(f"Agent version: {AGENT_VERSION}, TABLE_ID: {TABLE_ID}, PROJECT_ID: {PROJECT_ID}, DATASET_ID: {DATASET_ID}")

def get_table_list() -> List[str]:
    """
    Parse TABLE_ID environment variable as comma-separated list.
    
    Returns:
        List of table names with whitespace stripped
    """
    tables = [table.strip() for table in TABLE_ID.split(',')]
    return [t for t in tables if t]  # Filter out empty strings







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
    logging.info(f"Tool Call: parse_time_range(time_range='{time_range}')")
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
    logging.info("Tool Call: get_analysis_metadata()")
    from datetime import datetime
    
    tables = get_table_list()
    metadata = {
        "project_id": PROJECT_ID,
        "dataset": DATASET_ID,
        "tables": tables,  # Now returns list of tables
        "table_count": len(tables),
        "analyzer_version": AGENT_VERSION,
        "generated_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    return json.dumps(metadata, cls=AnalysisEncoder)


def verify_data_access() -> str:
    """
    Verifies BigQuery configuration and data access.
    
    Use this tool when you encounter "No data found" errors to check:
    1. If the configuration (Project, Dataset, Tables) is correct
    2. If the agent has permissions to access the table
    3. If the table actually contains data
    
    Returns:
        JSON string with configuration details and test query result
    """
    logging.info("Tool Call: verify_data_access()")
    tables = get_table_list()
    config = {
        "project_id": PROJECT_ID,
        "dataset": DATASET_ID,
        "tables": tables,
        "table_count": len(tables),
        "env_vars_loaded": {
            "PROJECT_ID": bool(PROJECT_ID),
            "DATASET_ID": bool(DATASET_ID),
            "TABLE_ID": bool(TABLE_ID)
        }
    }
    
    # Log configuration for visibility
    logging.info(f"[CONFIG] Verifying access with: Project={PROJECT_ID}, Dataset={DATASET_ID}, Tables={tables}")
    
    try:
        # Build query for all tables using UNION ALL
        table_queries = []
        for table in tables:
            table_query = f"""
            SELECT 
                '{table}' as table_name,
                COUNT(*) as total_rows,
                MIN(logging_time) as first_log,
                MAX(logging_time) as last_log
            FROM `{PROJECT_ID}.{DATASET_ID}.{table}`
            """
            table_queries.append(table_query)
        
        query = "\nUNION ALL\n".join(table_queries)
        
        df = execute_bigquery(query, timeout=30)
        
        if not df.empty:
            # Aggregate results across all tables
            total_rows = df['total_rows'].sum()
            first_log = df['first_log'].min()
            last_log = df['last_log'].max()
            
            result = {
                "status": "SUCCESS",
                "message": f"Successfully connected to {len(tables)} BigQuery table(s)",
                "total_rows": int(total_rows),
                "tables_detail": [
                    {
                        "table": row['table_name'],
                        "rows": int(row['total_rows']),
                        "first_log": row['first_log'].isoformat() if pd.notna(row['first_log']) else None,
                        "last_log": row['last_log'].isoformat() if pd.notna(row['last_log']) else None
                    }
                    for _, row in df.iterrows()
                ],
                "data_range": {
                    "start": first_log.isoformat() if pd.notna(first_log) else None,
                    "end": last_log.isoformat() if pd.notna(last_log) else None
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


def build_multi_table_source(where_clause: str, select_suffix: str = "") -> str:
    """
    Build a FROM clause that handles single or multiple tables.
    
    For single table: Returns simple FROM clause
    For multiple tables: Returns UNION ALL of all tables with optional alias
    
    Args:
        where_clause: WHERE conditions to apply (without WHERE keyword)
        select_suffix: Optional suffix to add after main table reference (e.g., "AS T")
        
    Returns:
        SQL fragment for FROM clause or UNION ALL subquery
        
    Example:
        Single table: `project.dataset.table` AS T
        Multiple tables: 
            (SELECT * FROM `project.dataset.table1` AS T WHERE ... 
             UNION ALL 
             SELECT * FROM `project.dataset.table2` AS T WHERE ...)
    """
    tables = get_table_list()
    
    if len(tables) == 1:
        # Single table - simple FROM with WHERE
        base_query = f"`{PROJECT_ID}.{DATASET_ID}.{tables[0]}` {select_suffix}"
        if where_clause:
            return f"(SELECT * FROM {base_query} WHERE {where_clause}) AS T"
        return base_query
    else:
        # Multiple tables - UNION ALL
        table_queries = []
        for table in tables:
            table_query = f"SELECT * FROM `{PROJECT_ID}.{DATASET_ID}.{table}` {select_suffix} WHERE {where_clause}"
            table_queries.append(table_query)
        
        return f"(\n{f' {chr(10)}    UNION ALL{chr(10)}'.join(table_queries)}\n  ) AS T"


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
    logging.info(f"Tool Call: get_overall_statistics(time_range='{time_range}', model_name='{model_name}', agent_name='{agent_name}')")
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
        
        # Build query for multiple tables using UNION ALL
        tables = get_table_list()
        
        if len(tables) == 1:
            # Single table - direct aggregation
            tables = get_table_list()
            
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
          `{PROJECT_ID}.{DATASET_ID}.{tables[0]}` AS T
        WHERE
          {where_clause}
        """
        else:
            # Multiple tables - union raw data then aggregate
            table_selects = []
            for table in tables:
                table_select = f"""
            SELECT
              CAST(JSON_EXTRACT_SCALAR(T.metadata, '$.request_latency') AS FLOAT64) / 1000 AS latency,
              SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.promptTokenCount) AS INT64) AS input_tokens,
              SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.candidatesTokenCount) AS INT64) AS output_tokens,
              SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.thoughtsTokenCount) AS INT64) AS thought_tokens,
              SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.totalTokenCount) AS INT64) AS total_tokens_row
            FROM
              `{PROJECT_ID}.{DATASET_ID}.{table}` AS T
            WHERE
              {where_clause}
            """
                table_selects.append(table_select)
            
            union_data = "\nUNION ALL\n".join(table_selects)
            
            query = f"""
        SELECT
          COUNT(*) as total_requests,
          AVG(latency) AS mean_latency,
          APPROX_QUANTILES(latency, 100)[OFFSET(50)] AS median_latency,
          APPROX_QUANTILES(latency, 100)[OFFSET(75)] AS p75_latency,
          APPROX_QUANTILES(latency, 100)[OFFSET(90)] AS p90_latency,
          APPROX_QUANTILES(latency, 100)[OFFSET(95)] AS p95_latency,
          APPROX_QUANTILES(latency, 100)[OFFSET(99)] AS p99_latency,
          APPROX_QUANTILES(latency, 1000)[OFFSET(999)] AS p999_latency,
          MIN(latency) AS min_latency,
          MAX(latency) AS max_latency,
          STDDEV(latency) AS std_latency,
          AVG(input_tokens) AS mean_input_tokens,
          AVG(output_tokens) AS mean_output_tokens,
          AVG(thought_tokens) AS mean_thought_tokens,
          AVG(total_tokens_row) AS mean_total_tokens,
          SUM(total_tokens_row) AS total_tokens
        FROM (
{union_data}
        )
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
    logging.info(f"Tool Call: get_latency_distribution(time_range='{time_range}', model_name='{model_name}', agent_name='{agent_name}')")
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
        
        # Build table source (handles single or multiple tables)
        tables = get_table_list()
        
        if len(tables) == 1:
            # Single table - use CTE approach
            tables = get_table_list()
            
            query = f"""
        WITH latency_data AS (
          SELECT
            CAST(JSON_EXTRACT_SCALAR(T.metadata, '$.request_latency') AS FLOAT64) / 1000 AS latency_seconds
          FROM
            `{PROJECT_ID}.{DATASET_ID}.{tables[0]}` AS T
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
        else:
            # Multiple tables - UNION ALL then aggregate
            table_selects = []
            for table in tables:
                table_select = f"""
          SELECT
            CAST(JSON_EXTRACT_SCALAR(T.metadata, '$.request_latency') AS FLOAT64) / 1000 AS latency_seconds
          FROM
            `{PROJECT_ID}.{DATASET_ID}.{table}` AS T
          WHERE
            {where_clause}
        """
                table_selects.append(table_select)
            
            union_data = "\nUNION ALL\n".join(table_selects)
            
            query = f"""
        WITH latency_data AS (
{union_data}
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
    logging.info(f"Tool Call: get_hourly_patterns(time_range='{time_range}', model_name='{model_name}', agent_name='{agent_name}')")
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
        
        # Build query for multiple tables using UNION ALL
        tables = get_table_list()
        
        if len(tables) == 1:
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
          `{PROJECT_ID}.{DATASET_ID}.{tables[0]}` AS T
        WHERE
          {where_clause}
        GROUP BY hour, day_of_week, day_type
        ORDER BY hour
        """
        else:
            # Multiple tables - union raw data then aggregate
            table_selects = []
            for table in tables:
                table_select = f"""
        SELECT
          EXTRACT(HOUR FROM T.logging_time) AS hour,
          EXTRACT(DAYOFWEEK FROM T.logging_time) AS day_of_week,
          CASE WHEN EXTRACT(DAYOFWEEK FROM T.logging_time) IN (1, 7) THEN 'weekend' ELSE 'working' END AS day_type,
          CAST(JSON_EXTRACT_SCALAR(T.metadata, '$.request_latency') AS FLOAT64) / 1000 AS latency
        FROM
          `{PROJECT_ID}.{DATASET_ID}.{table}` AS T
        WHERE
          {where_clause}
        """
                table_selects.append(table_select)
            
            union_data = "\nUNION ALL\n".join(table_selects)
            
            query = f"""
        SELECT
          hour,
          day_of_week,
          day_type,
          COUNT(*) as request_count,
          AVG(latency) AS avg_latency,
          MIN(latency) AS min_latency,
          MAX(latency) AS max_latency
        FROM (
{union_data}
        )
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
    logging.info(f"Tool Call: get_agent_comparison(time_range='{time_range}', model_name='{model_name}')")
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
        
        # Build query for multiple tables using UNION ALL
        tables = get_table_list()
        
        if len(tables) == 1:
            query = f"""
        SELECT
          COALESCE(JSON_VALUE(T.full_request.labels.adk_agent_name), 'unknown') AS agent_name,
          COUNT(*) as total_calls,
          AVG(CAST(JSON_EXTRACT_SCALAR(T.metadata, '$.request_latency') AS FLOAT64) / 1000) AS avg_latency,
          APPROX_QUANTILES(CAST(JSON_EXTRACT_SCALAR(T.metadata, '$.request_latency') AS FLOAT64) / 1000, 100)[OFFSET(95)] AS p95_latency,
          AVG(SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.promptTokenCount) AS INT64)) AS avg_input_tokens,
          AVG(SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.candidatesTokenCount) AS INT64)) AS avg_output_tokens,
          AVG(SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.thoughtsTokenCount) AS INT64)) AS avg_thought_tokens,
          AVG(SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.totalTokenCount) AS INT64)) AS avg_total_tokens,
          SUM(SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.totalTokenCount) AS INT64)) AS total_tokens,
          AVG(
             CASE 
                WHEN SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.candidatesTokenCount) AS INT64) > 0 
                THEN (CAST(JSON_EXTRACT_SCALAR(T.metadata, '$.request_latency') AS FLOAT64) / 1000) / SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.candidatesTokenCount) AS INT64)
                ELSE 0 
             END
          ) AS avg_tpot
        FROM
          `{PROJECT_ID}.{DATASET_ID}.{tables[0]}` AS T
        WHERE
          {where_clause}
        GROUP BY agent_name
        ORDER BY total_calls DESC
        """
        else:
            # Multiple tables - union raw data then aggregate
            table_selects = []
            for table in tables:
                table_select = f"""
        SELECT
          COALESCE(JSON_VALUE(T.full_request.labels.adk_agent_name), 'unknown') AS agent_name,
          CAST(JSON_EXTRACT_SCALAR(T.metadata, '$.request_latency') AS FLOAT64) / 1000 AS latency,
          SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.promptTokenCount) AS INT64) AS input_tokens,
          SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.candidatesTokenCount) AS INT64) AS output_tokens,
          SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.thoughtsTokenCount) AS INT64) AS thought_tokens,
          SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.totalTokenCount) AS INT64) AS total_tokens_row,
          CASE 
            WHEN SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.candidatesTokenCount) AS INT64) > 0 
            THEN (CAST(JSON_EXTRACT_SCALAR(T.metadata, '$.request_latency') AS FLOAT64) / 1000) / SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.candidatesTokenCount) AS INT64)
            ELSE 0 
          END AS tpot
        FROM
          `{PROJECT_ID}.{DATASET_ID}.{table}` AS T
        WHERE
          {where_clause}
        """
                table_selects.append(table_select)
            
            union_data = "\nUNION ALL\n".join(table_selects)
            
            query = f"""
        SELECT
          agent_name,
          COUNT(*) as total_calls,
          AVG(latency) AS avg_latency,
          APPROX_QUANTILES(latency, 100)[OFFSET(95)] AS p95_latency,
          AVG(input_tokens) AS avg_input_tokens,
          AVG(output_tokens) AS avg_output_tokens,
          AVG(thought_tokens) AS avg_thought_tokens,
          AVG(total_tokens_row) AS avg_total_tokens,
          SUM(total_tokens_row) AS total_tokens,
          AVG(tpot) AS avg_tpot
        FROM (
{union_data}
        )
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
            
            # Use explicit thought tokens if available, otherwise estimate
            avg_brain_tokens = 0
            if pd.notna(row['avg_thought_tokens']):
                 avg_brain_tokens = row['avg_thought_tokens']
            elif pd.notna(row['avg_total_tokens']) and pd.notna(row['avg_input_tokens']) and pd.notna(row['avg_output_tokens']):
                 avg_brain_tokens = max(0, row['avg_total_tokens'] - row['avg_input_tokens'] - row['avg_output_tokens'])
            
            # Thought ratio
            thought_ratio = 0
            if pd.notna(row['avg_output_tokens']) and row['avg_output_tokens'] > 0:
                thought_ratio = avg_brain_tokens / row['avg_output_tokens']
            
            agents.append({
                "agent_name": row['agent_name'],
                "total_calls": int(row['total_calls']),
                "avg_latency": float(row['avg_latency']),
                "p95_latency": float(row['p95_latency']) if pd.notna(row['p95_latency']) else None,
                "avg_input_tokens": float(row['avg_input_tokens']) if pd.notna(row['avg_input_tokens']) else None,
                "avg_output_tokens": float(row['avg_output_tokens']) if pd.notna(row['avg_output_tokens']) else None,
                "avg_thought_tokens": float(avg_brain_tokens),
                "thought_output_ratio": float(thought_ratio),
                "avg_tpot": float(row['avg_tpot']) if pd.notna(row['avg_tpot']) else None,
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


def get_model_comparison(
    time_range: str = "24h",
    agent_name: Optional[str] = None
) -> str:
    """
    Compare performance across different models.
    
    Extracts model name from the model field (e.g., "publishers/google/models/gemini-2.5-pro")
    and returns per-model statistics including calls, latency, and token usage.
    
    Args:
        time_range: Time range to analyze
        agent_name: Filter by specific agent (optional)
        
    Returns:
        JSON string with per-model performance comparison
    """
    logging.info(f"Tool Call: get_model_comparison(time_range='{time_range}', agent_name='{agent_name}')")
    try:
        time_range_dict = json.loads(parse_time_range(time_range))
        start_time, end_time = time_range_dict['start_date'], time_range_dict['end_date']
        
        where_clauses = [
            f"T.logging_time BETWEEN '{start_time}' AND '{end_time}'",
            "T.full_request IS NOT NULL",
            "T.full_response IS NOT NULL",
            "T.model IS NOT NULL",
            "JSON_VALUE(T.metadata.request_latency) IS NOT NULL"
        ]
        
        if agent_name:
            where_clauses.append(f"JSON_VALUE(T.full_request.labels.adk_agent_name) = '{agent_name}'")
        
        where_clause = " AND ".join(where_clauses)
        
        # Build query for multiple tables using UNION ALL
        tables = get_table_list()
        
        if len(tables) == 1:
            # Single table - direct aggregation
            tables = get_table_list()
            
            query = f"""
        SELECT
          SPLIT(T.model, '/')[SAFE_OFFSET(1)] AS publisher,
          SPLIT(T.model, '/')[SAFE_OFFSET(3)] AS model_name,
          T.model AS full_model_path,
          COUNT(*) as total_calls,
          AVG(CAST(JSON_EXTRACT_SCALAR(T.metadata, '$.request_latency') AS FLOAT64) / 1000) AS avg_latency,
          APPROX_QUANTILES(CAST(JSON_EXTRACT_SCALAR(T.metadata, '$.request_latency') AS FLOAT64) / 1000, 100)[OFFSET(95)] AS p95_latency,
          AVG(SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.promptTokenCount) AS INT64)) AS avg_input_tokens,
          AVG(SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.candidatesTokenCount) AS INT64)) AS avg_output_tokens,
          AVG(SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.thoughtsTokenCount) AS INT64)) AS avg_thought_tokens,
          AVG(SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.totalTokenCount) AS INT64)) AS avg_total_tokens,
          SUM(SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.totalTokenCount) AS INT64)) AS total_tokens,
          AVG(
             CASE 
                WHEN SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.candidatesTokenCount) AS INT64) > 0 
                THEN (CAST(JSON_EXTRACT_SCALAR(T.metadata, '$.request_latency') AS FLOAT64) / 1000) / SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.candidatesTokenCount) AS INT64)
                ELSE 0 
             END
          ) AS avg_tpot
        FROM
          `{PROJECT_ID}.{DATASET_ID}.{tables[0]}` AS T
        WHERE
          {where_clause}
        GROUP BY publisher, model_name, full_model_path
        ORDER BY total_calls DESC
        """
        else:
            # Multiple tables - union raw data then aggregate
            table_selects = []
            for table in tables:
                table_select = f"""
            SELECT
              T.model,
              CAST(JSON_EXTRACT_SCALAR(T.metadata, '$.request_latency') AS FLOAT64) / 1000 AS latency,
              SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.promptTokenCount) AS INT64) AS input_tokens,
              SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.candidatesTokenCount) AS INT64) AS output_tokens,
              SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.thoughtsTokenCount) AS INT64) AS thought_tokens,
              SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.totalTokenCount) AS INT64) AS total_tokens,
              CASE 
                WHEN SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.candidatesTokenCount) AS INT64) > 0 
                THEN (CAST(JSON_EXTRACT_SCALAR(T.metadata, '$.request_latency') AS FLOAT64) / 1000) / SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.candidatesTokenCount) AS INT64)
                ELSE 0 
              END AS tpot
            FROM
              `{PROJECT_ID}.{DATASET_ID}.{table}` AS T
            WHERE
              {where_clause}
            """
                table_selects.append(table_select)
            
            union_data = "\nUNION ALL\n".join(table_selects)
            
            query = f"""
        SELECT
          SPLIT(model, '/')[SAFE_OFFSET(1)] AS publisher,
          SPLIT(model, '/')[SAFE_OFFSET(3)] AS model_name,
          model AS full_model_path,
          COUNT(*) as total_calls,
          AVG(latency) AS avg_latency,
          APPROX_QUANTILES(latency, 100)[OFFSET(95)] AS p95_latency,
          AVG(input_tokens) AS avg_input_tokens,
          AVG(output_tokens) AS avg_output_tokens,
          AVG(thought_tokens) AS avg_thought_tokens,
          AVG(total_tokens) AS avg_total_tokens,
          SUM(total_tokens) AS total_tokens_sum,
          AVG(tpot) AS avg_tpot
        FROM (
{union_data}
        )
        GROUP BY publisher, model_name, full_model_path
        ORDER BY total_calls DESC
        """
        
        df = execute_bigquery(query)
        
        if df.empty:
            return json.dumps({"error": "No data found"})
        
        models = []
        for _, row in df.iterrows():
            # Calculate efficiency score (lower is better): latency per 1000 tokens
            efficiency = None
            if pd.notna(row['avg_input_tokens']) and row['avg_input_tokens'] > 0:
                efficiency = float(row['avg_latency']) / (float(row['avg_input_tokens']) / 1000)
            
            # Use explicit thought tokens if available, otherwise estimate
            avg_brain_tokens = 0
            if pd.notna(row['avg_thought_tokens']):
                 avg_brain_tokens = row['avg_thought_tokens']
            elif pd.notna(row['avg_total_tokens']) and pd.notna(row['avg_input_tokens']) and pd.notna(row['avg_output_tokens']):
                 avg_brain_tokens = max(0, row['avg_total_tokens'] - row['avg_input_tokens'] - row['avg_output_tokens'])
            
            # Thought ratio
            thought_ratio = 0
            if pd.notna(row['avg_output_tokens']) and row['avg_output_tokens'] > 0:
                thought_ratio = avg_brain_tokens / row['avg_output_tokens']
            
            model_name_clean = row['model_name'] if pd.notna(row['model_name']) else row['full_model_path']
            publisher = row['publisher'] if pd.notna(row['publisher']) else 'unknown'
            
            models.append({
                "model_name": model_name_clean,
                "publisher": publisher,
                "full_model_path": row['full_model_path'],
                "total_calls": int(row['total_calls']),
                "avg_latency": float(row['avg_latency']),
                "p95_latency": float(row['p95_latency']) if pd.notna(row['p95_latency']) else None,
                "avg_input_tokens": float(row['avg_input_tokens']) if pd.notna(row['avg_input_tokens']) else None,
                "avg_output_tokens": float(row['avg_output_tokens']) if pd.notna(row['avg_output_tokens']) else None,
                "avg_thought_tokens": float(avg_brain_tokens),
                "thought_output_ratio": float(thought_ratio),
                "avg_tpot": float(row['avg_tpot']) if pd.notna(row['avg_tpot']) else None,
                "total_tokens": int(row['total_tokens_sum'] if 'total_tokens_sum' in row else row['total_tokens']) if pd.notna(row.get('total_tokens_sum', row.get('total_tokens'))) else None,
                "efficiency_score": efficiency
            })
        
        # Rank by efficiency
        models_with_efficiency = [m for m in models if m['efficiency_score'] is not None]
        if models_with_efficiency:
            models_with_efficiency.sort(key=lambda x: x['efficiency_score'])
            best_model = models_with_efficiency[0]['model_name']
            worst_model = models_with_efficiency[-1]['model_name']
        else:
            best_model = None
            worst_model = None
        
        # Rank by latency
        models_sorted_by_latency = sorted(models, key=lambda x: x['avg_latency'])
        fastest_model = models_sorted_by_latency[0]['model_name'] if models_sorted_by_latency else None
        slowest_model = models_sorted_by_latency[-1]['model_name'] if models_sorted_by_latency else None
        
        result = {
            "metadata": {
                "time_range": f"{start_time} to {end_time}",
                "total_models": len(models)
            },
            "models": models,
            "insights": {
                "most_efficient_model": best_model,
                "least_efficient_model": worst_model,
                "fastest_model": fastest_model,
                "slowest_model": slowest_model,
                "most_active_model": models[0]['model_name'] if models else None
            },
            "summary": f"Analyzed {len(models)} models. Most active: {models[0]['model_name']} with {models[0]['total_calls']} calls. Fastest: {fastest_model} ({models_sorted_by_latency[0]['avg_latency']:.2f}s avg)" if models else "No data"
        }
        
        return json.dumps(result, cls=AnalysisEncoder)
        
    except Exception as e:
        logging.error(f"Error in get_model_comparison: {str(e)}")
        return json.dumps({"error": str(e)})


def get_agent_model_matrix(
    time_range: str = "24h"
) -> str:
    """
    Get performance matrix for all agent-model combinations.
    
    This provides a comprehensive view of how each agent performs with each model,
    enabling detection of:
    - Agent-specific model preferences
    - Model switching impacts per agent
    - Agent-model combinations that are problematic
    
    Args:
        time_range: Time range to analyze
        
    Returns:
        JSON string with agent-model performance matrix
    """
    logging.info(f"Tool Call: get_agent_model_matrix(time_range='{time_range}')")
    try:
        time_range_dict = json.loads(parse_time_range(time_range))
        start_time, end_time = time_range_dict['start_date'], time_range_dict['end_date']
        
        where_clauses = [
            f"T.logging_time BETWEEN '{start_time}' AND '{end_time}'",
            "T.full_request IS NOT NULL",
            "T.full_response IS NOT NULL",
            "T.model IS NOT NULL",
            "JSON_VALUE(T.metadata.request_latency) IS NOT NULL"
        ]
        
        where_clause = " AND ".join(where_clauses)
        
        # Build query for multiple tables using UNION ALL
        tables = get_table_list()
        
        if len(tables) == 1:
            # Single table - direct aggregation
            tables = get_table_list()
            
            query = f"""
        SELECT
          COALESCE(JSON_VALUE(T.full_request.labels.adk_agent_name), 'unknown') AS agent_name,
          SPLIT(T.model, '/')[SAFE_OFFSET(1)] AS publisher,
          SPLIT(T.model, '/')[SAFE_OFFSET(3)] AS model_name,
          T.model AS full_model_path,
          COUNT(*) as total_calls,
          AVG(CAST(JSON_EXTRACT_SCALAR(T.metadata, '$.request_latency') AS FLOAT64) / 1000) AS avg_latency,
          APPROX_QUANTILES(CAST(JSON_EXTRACT_SCALAR(T.metadata, '$.request_latency') AS FLOAT64) / 1000, 100)[OFFSET(95)] AS p95_latency,
          AVG(SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.promptTokenCount) AS INT64)) AS avg_input_tokens,
          AVG(SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.candidatesTokenCount) AS INT64)) AS avg_output_tokens,
          AVG(SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.thoughtsTokenCount) AS INT64)) AS avg_thought_tokens,
          AVG(
             CASE 
                WHEN SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.candidatesTokenCount) AS INT64) > 0 
                THEN (CAST(JSON_EXTRACT_SCALAR(T.metadata, '$.request_latency') AS FLOAT64) / 1000) / SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.candidatesTokenCount) AS INT64)
                ELSE 0 
             END
          ) AS avg_tpot
        FROM
          `{PROJECT_ID}.{DATASET_ID}.{tables[0]}` AS T
        WHERE
          {where_clause}
        GROUP BY agent_name, publisher, model_name, full_model_path
        ORDER BY agent_name, total_calls DESC
        """
        else:
            # Multiple tables - union raw data then aggregate
            table_selects = []
            for table in tables:
                table_select = f"""
            SELECT
              COALESCE(JSON_VALUE(T.full_request.labels.adk_agent_name), 'unknown') AS agent_name,
              T.model,
              CAST(JSON_EXTRACT_SCALAR(T.metadata, '$.request_latency') AS FLOAT64) / 1000 AS latency,
              SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.promptTokenCount) AS INT64) AS input_tokens,
              SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.candidatesTokenCount) AS INT64) AS output_tokens,
              SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.thoughtsTokenCount) AS INT64) AS thought_tokens,
              CASE 
                WHEN SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.candidatesTokenCount) AS INT64) > 0 
                THEN (CAST(JSON_EXTRACT_SCALAR(T.metadata, '$.request_latency') AS FLOAT64) / 1000) / SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.candidatesTokenCount) AS INT64)
                ELSE 0 
              END AS tpot
            FROM
              `{PROJECT_ID}.{DATASET_ID}.{table}` AS T
            WHERE
              {where_clause}
            """
                table_selects.append(table_select)
            
            union_data = "\nUNION ALL\n".join(table_selects)
            
            query = f"""
        SELECT
          agent_name,
          SPLIT(model, '/')[SAFE_OFFSET(1)] AS publisher,
          SPLIT(model, '/')[SAFE_OFFSET(3)] AS model_name,
          model AS full_model_path,
          COUNT(*) as total_calls,
          AVG(latency) AS avg_latency,
          APPROX_QUANTILES(latency, 100)[OFFSET(95)] AS p95_latency,
          AVG(input_tokens) AS avg_input_tokens,
          AVG(output_tokens) AS avg_output_tokens,
          AVG(thought_tokens) AS avg_thought_tokens,
          AVG(tpot) AS avg_tpot
        FROM (
{union_data}
        )
        GROUP BY agent_name, publisher, model_name, full_model_path
        ORDER BY agent_name, total_calls DESC
        """
        
        df = execute_bigquery(query)
        
        if df.empty:
            return json.dumps({"error": "No data found"})
        
        # Build matrix structure: agent -> models
        agent_model_matrix = {}
        all_models = set()
        all_agents = set()
        
        for _, row in df.iterrows():
            agent = row['agent_name']
            model_name_clean = row['model_name'] if pd.notna(row['model_name']) else row['full_model_path']
            
            all_agents.add(agent)
            all_models.add(model_name_clean)
            
            if agent not in agent_model_matrix:
                agent_model_matrix[agent] = {}
            
            agent_model_matrix[agent][model_name_clean] = {
                "total_calls": int(row['total_calls']),
                "avg_latency": float(row['avg_latency']),
                "p95_latency": float(row['p95_latency']) if pd.notna(row['p95_latency']) else None,
                "avg_input_tokens": float(row['avg_input_tokens']) if pd.notna(row['avg_input_tokens']) else None,
                "avg_output_tokens": float(row['avg_output_tokens']) if pd.notna(row['avg_output_tokens']) else None,
                "avg_thought_tokens": float(row['avg_thought_tokens']) if pd.notna(row['avg_thought_tokens']) else None,
                "avg_tpot": float(row['avg_tpot']) if pd.notna(row['avg_tpot']) else None,
                "publisher": row['publisher'] if pd.notna(row['publisher']) else 'unknown'
            }
        
        # Find insights
        slowest_combo = None
        fastest_combo = None
        min_latency = float('inf')
        max_latency = 0
        
        for agent, models in agent_model_matrix.items():
            for model, stats in models.items():
                latency = stats['avg_latency']
                if latency < min_latency:
                    min_latency = latency
                    fastest_combo = {"agent": agent, "model": model, "latency": latency}
                if latency > max_latency:
                    max_latency = latency
                    slowest_combo = {"agent": agent, "model": model, "latency": latency}
        
        # Check for model switching within agents
        agents_with_multiple_models = {
            agent: list(models.keys()) 
            for agent, models in agent_model_matrix.items() 
            if len(models) > 1
        }
        
        result = {
            "metadata": {
                "time_range": f"{start_time} to {end_time}",
                "total_agents": len(all_agents),
                "total_models": len(all_models),
                "total_combinations": sum(len(models) for models in agent_model_matrix.values())
            },
            "matrix": agent_model_matrix,
            "agents_with_multiple_models": agents_with_multiple_models,
            "insights": {
                "slowest_combination": slowest_combo,
                "fastest_combination": fastest_combo,
                "model_switching_detected": len(agents_with_multiple_models) > 0,
                "agents_switching_models": list(agents_with_multiple_models.keys()) if agents_with_multiple_models else []
            },
            "summary": f"Analyzed {len(all_agents)} agents × {len(all_models)} models = {sum(len(models) for models in agent_model_matrix.values())} combinations. {len(agents_with_multiple_models)} agents use multiple models."
        }
        
        return json.dumps(result, cls=AnalysisEncoder)
        
    except Exception as e:
        logging.error(f"Error in get_agent_model_matrix: {str(e)}")
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
    logging.info(f"Tool Call: get_token_correlation(time_range='{time_range}', model_name='{model_name}', agent_name='{agent_name}')")
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
        
        # Build query for multiple tables using UNION ALL
        tables = get_table_list()
        
        if len(tables) == 1:
            query = f"""
        SELECT
          CAST(JSON_EXTRACT_SCALAR(T.metadata, '$.request_latency') AS FLOAT64) / 1000 AS latency,
          SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.promptTokenCount) AS INT64) AS input_tokens,
          SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.candidatesTokenCount) AS INT64) AS output_tokens,
          SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.thoughtsTokenCount) AS INT64) AS thought_tokens
        FROM
          `{PROJECT_ID}.{DATASET_ID}.{tables[0]}` AS T
        WHERE
          {where_clause}
        LIMIT 1000
        """
        else:
            # Multiple tables - union data
            table_selects = []
            for table in tables:
                table_select = f"""
        SELECT
          CAST(JSON_EXTRACT_SCALAR(T.metadata, '$.request_latency') AS FLOAT64) / 1000 AS latency,
          SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.promptTokenCount) AS INT64) AS input_tokens,
          SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.candidatesTokenCount) AS INT64) AS output_tokens,
          SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.thoughtsTokenCount) AS INT64) AS thought_tokens
        FROM
          `{PROJECT_ID}.{DATASET_ID}.{table}` AS T
        WHERE
          {where_clause}
        """
                table_selects.append(table_select)
            
            union_data = "\nUNION ALL\n".join(table_selects)
            
            query = f"""
        SELECT * FROM (
{union_data}
        )
        LIMIT 1000
        """
        
        df = execute_bigquery(query)
        
        if df.empty or len(df) < 2:
            return json.dumps({"error": "Insufficient data for correlation analysis"})
        
        # Calculate correlations
        corr_input = df['latency'].corr(df['input_tokens']) if df['input_tokens'].notna().sum() > 1 else None
        corr_output = df['latency'].corr(df['output_tokens']) if df['output_tokens'].notna().sum() > 1 else None
        corr_thought = df['latency'].corr(df['thought_tokens']) if df['thought_tokens'].notna().sum() > 1 else None
        
        # Calculate combined Output + Thought correlation (User Request)
        df['combined_tokens'] = df['output_tokens'].fillna(0) + df['thought_tokens'].fillna(0)
        corr_combined = df['latency'].corr(df['combined_tokens']) if df['combined_tokens'].notna().sum() > 1 else None
        
        # Sample data points for visualization (limit to 100 for JSON size)
        sample_df = df.sample(min(100, len(df)))
        scatter_data = []
        for _, row in sample_df.iterrows():
            scatter_data.append({
                "latency": float(row['latency']),
                "input_tokens": int(row['input_tokens']) if pd.notna(row['input_tokens']) else None,
                "output_tokens": int(row['output_tokens']) if pd.notna(row['output_tokens']) else None,
                "thought_tokens": int(row['thought_tokens']) if pd.notna(row['thought_tokens']) else None,
                "combined_tokens": int(row['combined_tokens'])
            })
        
        result = {
            "metadata": {
                "time_range": f"{start_time} to {end_time}",
                "sample_size": len(df)
            },
            "correlations": {
                "latency_vs_input_tokens": float(corr_input) if corr_input is not None else None,
                "latency_vs_output_tokens": float(corr_output) if corr_output is not None else None,
                "latency_vs_thought_tokens": float(corr_thought) if corr_thought is not None else None,
                "latency_vs_output_plus_thought_tokens": float(corr_combined) if corr_combined is not None else None
            },
            "scatter_data": scatter_data,
            "summary": f"Input tokens correlation: {corr_input:.3f}, Output tokens correlation: {corr_output:.3f}, Output+Thought correlation: {corr_combined:.3f}" if corr_input and corr_output and corr_combined else "Insufficient data"
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
    logging.info(f"Tool Call: get_outlier_analysis(time_range='{time_range}', model_name='{model_name}', agent_name='{agent_name}', threshold_std={threshold_std})")
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
        
        # Build query for multiple tables using UNION ALL
        tables = get_table_list()
        
        # First get mean and std
        if len(tables) == 1:
            tables = get_table_list()
            
            stats_query = f"""
        SELECT
          AVG(CAST(JSON_EXTRACT_SCALAR(T.metadata, '$.request_latency') AS FLOAT64) / 1000) AS mean_latency,
          STDDEV(CAST(JSON_EXTRACT_SCALAR(T.metadata, '$.request_latency') AS FLOAT64) / 1000) AS std_latency
        FROM
          `{PROJECT_ID}.{DATASET_ID}.{tables[0]}` AS T
        WHERE
          {where_clause}
        """
        else:
            table_selects = []
            for table in tables:
                table_select = f"""
        SELECT
          CAST(JSON_EXTRACT_SCALAR(T.metadata, '$.request_latency') AS FLOAT64) / 1000 AS latency
        FROM
          `{PROJECT_ID}.{DATASET_ID}.{table}` AS T
        WHERE
          {where_clause}
        """
                table_selects.append(table_select)
            
            union_data = "\nUNION ALL\n".join(table_selects)
            
            tables = get_table_list()
            
            stats_query = f"""
        SELECT
          AVG(latency) AS mean_latency,
          STDDEV(latency) AS std_latency
        FROM (
{union_data}
        )
        """
        
        stats_df = execute_bigquery(stats_query)
        mean_latency = float(stats_df.iloc[0]['mean_latency'])
        std_latency = float(stats_df.iloc[0]['std_latency'])
        threshold = mean_latency + (threshold_std * std_latency)
        
        # Now get outliers
        if len(tables) == 1:
            tables = get_table_list()
            
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
          `{PROJECT_ID}.{DATASET_ID}.{tables[0]}` AS T
        WHERE
          {where_clause}
          AND CAST(JSON_EXTRACT_SCALAR(T.metadata, '$.request_latency') AS FLOAT64) / 1000 > {threshold}
        ORDER BY latency DESC
        LIMIT 50
        """
        else:
            table_selects = []
            for table in tables:
                table_select = f"""
        SELECT
          CAST(T.request_id AS STRING) AS request_id,
          T.logging_time,
          CAST(JSON_EXTRACT_SCALAR(T.metadata, '$.request_latency') AS FLOAT64) / 1000 AS latency,
          COALESCE(JSON_VALUE(T.full_request.labels.adk_agent_name), 'unknown') AS agent_name,
          SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.promptTokenCount) AS INT64) AS input_tokens,
          SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.candidatesTokenCount) AS INT64) AS output_tokens,
          SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.thoughtsTokenCount) AS INT64) AS thought_tokens
        FROM
          `{PROJECT_ID}.{DATASET_ID}.{table}` AS T
        WHERE
          {where_clause}
          AND CAST(JSON_EXTRACT_SCALAR(T.metadata, '$.request_latency') AS FLOAT64) / 1000 > {threshold}
        """
                table_selects.append(table_select)
            
            union_data = "\nUNION ALL\n".join(table_selects)
            
            outliers_query = f"""
        SELECT * FROM (
{union_data}
        )
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
    logging.info(f"Tool Call: get_slowest_queries(num_queries={num_queries}, time_range='{time_range}', model_name='{model_name}', agent_name='{agent_name}')")
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
        
        # Build multi-table source
        union_source = build_multi_table_source(where_clause, select_suffix="AS T")
        
        query = f"""
        SELECT
          CAST(request_id AS STRING) AS request_id,
          logging_time,
          CAST(JSON_EXTRACT_SCALAR(metadata, '$.request_latency') AS FLOAT64) / 1000 AS latency,
          model,
          COALESCE(JSON_VALUE(full_request.labels.adk_agent_name), 'unknown') AS agent_name,
          SAFE_CAST(JSON_VALUE(full_response.usageMetadata.promptTokenCount) AS INT64) AS input_tokens,
          SAFE_CAST(JSON_VALUE(full_response.usageMetadata.candidatesTokenCount) AS INT64) AS output_tokens,
          SAFE_CAST(JSON_VALUE(full_response.usageMetadata.thoughtsTokenCount) AS INT64) AS thought_tokens,
          SAFE_CAST(JSON_VALUE(full_response.usageMetadata.totalTokenCount) AS INT64) AS total_tokens,
          -- Extract last user message text (robustly via ARRAY/LIMIT)
          ARRAY(
            SELECT 
              (SELECT STRING_AGG(JSON_VALUE(p, '$.text'), ' ') FROM UNNEST(JSON_QUERY_ARRAY(c, '$.parts')) AS p)
            FROM UNNEST(JSON_QUERY_ARRAY(full_request, '$.contents')) AS c WITH OFFSET AS off
            WHERE JSON_VALUE(c, '$.role') = 'user'
            ORDER BY off DESC
            LIMIT 1
          )[SAFE_OFFSET(0)] AS last_user_message
        FROM 
            {union_source}
        ORDER BY latency DESC
        LIMIT {num_queries}
        """
        
        logging.info(f"Generated SQL for get_slowest_queries:\n{query}")
        print(f"DEBUG: generated query: {query}") # Verbose print as requested
        
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
    logging.info(f"Tool Call: get_query_details(request_id='{request_id}')")
    try:
        tables = get_table_list()
        
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
          `{PROJECT_ID}.{DATASET_ID}.{tables[0]}` AS T
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
    logging.info(f"Tool Call: get_concurrent_request_impact(time_range='{time_range}', model_name='{model_name}', bucket_size={bucket_size})")
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
        
        tables = get_table_list()
        
        query = f"""
        WITH bucketed_requests AS (
          SELECT
            TIMESTAMP_TRUNC(T.logging_time, SECOND, 'UTC') AS bucket_start,
            TIMESTAMP_ADD(TIMESTAMP_TRUNC(T.logging_time, SECOND, 'UTC'), INTERVAL {bucket_size} SECOND) AS bucket_end,
            CAST(JSON_EXTRACT_SCALAR(T.metadata, '$.request_latency') AS FLOAT64) / 1000 AS latency
          FROM
            `{PROJECT_ID}.{DATASET_ID}.{tables[0]}` AS T
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
    logging.info(f"Tool Call: detect_performance_degradation(time_range='{time_range}', model_name='{model_name}', window_size={window_size})")
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
        
        tables = get_table_list()
        
        query = f"""
        SELECT
          TIMESTAMP_TRUNC(T.logging_time, HOUR) AS hour,
          AVG(CAST(JSON_EXTRACT_SCALAR(T.metadata, '$.request_latency') AS FLOAT64) / 1000) AS avg_latency,
          COUNT(*) as request_count
        FROM
          `{PROJECT_ID}.{DATASET_ID}.{tables[0]}` AS T
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
    logging.info(f"Tool Call: get_cost_analysis(time_range='{time_range}', model_name='{model_name}', agent_name='{agent_name}')")
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
        
        tables = get_table_list()
        
        query = f"""
        SELECT
          COALESCE(JSON_VALUE(T.full_request.labels.adk_agent_name), 'unknown') AS agent_name,
          COUNT(*) as total_requests,
          SUM(SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.promptTokenCount) AS INT64)) AS total_input_tokens,
          SUM(SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.candidatesTokenCount) AS INT64)) AS total_output_tokens,
          SUM(SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.totalTokenCount) AS INT64)) AS total_tokens,
          AVG(SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.totalTokenCount) AS INT64)) AS avg_tokens_per_request
        FROM
          `{PROJECT_ID}.{DATASET_ID}.{tables[0]}` AS T
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
    logging.info(f"Tool Call: compare_time_periods(period1='{period1}', period2='{period2}', model_name='{model_name}', agent_name='{agent_name}')")
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
            
            tables = get_table_list()
            
            query = f"""
            SELECT
              COUNT(*) as total_requests,
              AVG(CAST(JSON_EXTRACT_SCALAR(T.metadata, '$.request_latency') AS FLOAT64) / 1000) AS mean_latency,
              APPROX_QUANTILES(CAST(JSON_EXTRACT_SCALAR(T.metadata, '$.request_latency') AS FLOAT64) / 1000, 100)[OFFSET(95)] AS p95_latency,
              AVG(SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.totalTokenCount) AS INT64)) AS avg_tokens
            FROM
              `{PROJECT_ID}.{DATASET_ID}.{tables[0]}` AS T
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
    logging.info(f"Tool Call: cluster_slow_queries(num_queries={num_queries}, time_range='{time_range}', model_name='{model_name}', agent_name='{agent_name}')")
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
        
        tables = get_table_list()
        
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
          `{PROJECT_ID}.{DATASET_ID}.{tables[0]}` AS T
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
            total_t = row['total_tokens'] or (input_t + output_t)
            latency = row['latency']
            
            # NEW: Detect "Anomalous Inefficiency"
            # Normal token count (<500) but unexpectedly high latency (>10s)
            if total_t < 500 and latency > 10:
                return "anomalous_inefficiency"
                
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
    logging.info(f"Tool Call: analyze_correlation_detailed(time_range='{time_range}', model_name='{model_name}', agent_name='{agent_name}')")
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
        
        tables = get_table_list()
        
        query = f"""
        SELECT
          CAST(JSON_EXTRACT_SCALAR(T.metadata, '$.request_latency') AS FLOAT64) / 1000 AS latency,
          SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.promptTokenCount) AS INT64) AS input_tokens,
          SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.candidatesTokenCount) AS INT64) AS output_tokens,
          SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.thoughtsTokenCount) AS INT64) AS thought_tokens,
          SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.totalTokenCount) AS INT64) AS total_tokens
        FROM
          `{PROJECT_ID}.{DATASET_ID}.{tables[0]}` AS T
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
    logging.info(f"Tool Call: fetch_slow_queries(num_records={num_records}, agent_name='{agent_name}')")
    logging.info(f"[PROGRESS] Starting fetch of {num_records} slowest queries")
    try:
        where_clause = "T.full_request IS NOT NULL AND T.full_response IS NOT NULL"
        if agent_name:
            where_clause += f" AND JSON_VALUE(T.full_request.labels.adk_agent_name) = '{agent_name}'"

        tables = get_table_list()
        
        if len(tables) == 1:
            query = f"""
        SELECT
          CAST(T.request_id AS STRING) AS request_id,
          ROUND(SAFE_CAST(JSON_VALUE(T.metadata.request_latency) AS FLOAT64) / 1000.0, 2) AS request_latency_seconds
        FROM
          `{PROJECT_ID}.{DATASET_ID}.{tables[0]}` AS T
        WHERE
          {where_clause}
        ORDER BY
          request_latency_seconds DESC
        LIMIT {num_records}
        """
        else:
            table_selects = []
            for table in tables:
                table_select = f"""
        SELECT
          CAST(T.request_id AS STRING) AS request_id,
          ROUND(SAFE_CAST(JSON_VALUE(T.metadata.request_latency) AS FLOAT64) / 1000.0, 2) AS request_latency_seconds
        FROM
          `{PROJECT_ID}.{DATASET_ID}.{table}` AS T
        WHERE
          {where_clause}
        """
                table_selects.append(table_select)
            
            union_data = "\nUNION ALL\n".join(table_selects)
            
            query = f"""
        SELECT * FROM (
{union_data}
        )
        ORDER BY request_latency_seconds DESC
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
    logging.info(f"Tool Call: fetch_single_query(request_id='{request_id}')")
    logging.info(f"[PROGRESS] Starting fetch for query {request_id}")
    try:
        tables = get_table_list()
        
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
          `{PROJECT_ID}.{DATASET_ID}.{tables[0]}` AS T
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
    logging.info(f"Tool Call: fetch_slow_queries_batch(num_queries={num_queries}, time_range='{time_range}', model_name='{model_name}', agent_name='{agent_name}')")
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
        
        tables = get_table_list()
        
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
          `{PROJECT_ID}.{DATASET_ID}.{tables[0]}` AS T
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
    logging.info(f"Tool Call: fetch_fastest_queries(num_queries={num_queries}, time_range='{time_range}', model_name='{model_name}', agent_name='{agent_name}')")
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
        
        tables = get_table_list()
        
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
          `{PROJECT_ID}.{DATASET_ID}.{tables[0]}` AS T
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
    logging.info(f"Tool Call: get_token_velocity(time_range='{time_range}', model_name='{model_name}', agent_name='{agent_name}')")
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
        
        tables = get_table_list()
        
        query = f"""
        SELECT
          CAST(JSON_EXTRACT_SCALAR(T.metadata, '$.request_latency') AS FLOAT64) / 1000 AS latency,
          SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.candidatesTokenCount) AS INT64) AS output_tokens
        FROM
          `{PROJECT_ID}.{DATASET_ID}.{tables[0]}` AS T
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
    logging.info(f"Tool Call: analyze_request_queuing(time_range='{time_range}', model_name='{model_name}', agent_name='{agent_name}', burst_window_seconds={burst_window_seconds})")
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
        tables = get_table_list()
        
        query = f"""
        WITH bursts AS (
          SELECT
            TIMESTAMP_TRUNC(T.logging_time, SECOND) AS burst_time,
            COUNT(*) as request_count,
            AVG(CAST(JSON_EXTRACT_SCALAR(T.metadata, '$.request_latency') AS FLOAT64) / 1000) AS avg_latency,
            MAX(CAST(JSON_EXTRACT_SCALAR(T.metadata, '$.request_latency') AS FLOAT64) / 1000) AS max_latency
          FROM
            `{PROJECT_ID}.{DATASET_ID}.{tables[0]}` AS T
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
    logging.info(f"Tool Call: check_kpi_compliance(time_range='{time_range}', mean_latency_target={mean_latency_target}, p95_latency_target={p95_latency_target}, agent_name='{agent_name}')")
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

        # Reuse get_overall_statistics to get current global metrics
        stats_json = get_overall_statistics(time_range=time_range, agent_name=agent_name)
        stats = json.loads(stats_json)
        
        if "error" in stats:
            return stats_json
            
        current_mean = stats['latency']['mean']
        current_p95 = stats['latency']['p95']
        
        # Global Compliance
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

        # Per-Agent Compliance (if no specific agent filter is applied)
        per_agent_compliance = []
        if agent_name is None:
            # Use get_agent_comparison to get per-agent stats
            agent_stats_json = get_agent_comparison(time_range=time_range)
            agent_stats = json.loads(agent_stats_json)
            
            if "agents" in agent_stats:
                for agent in agent_stats["agents"]:
                    a_name = agent["agent_name"]
                    a_mean = agent["avg_latency"]
                    a_p95 = agent["p95_latency"] if agent["p95_latency"] is not None else 0.0
                    
                    per_agent_compliance.append({
                        "agent_name": a_name,
                        "mean_latency": a_mean,
                        "p95_latency": a_p95,
                        "status": "PASS" if (a_mean <= mean_latency_target and a_p95 <= p95_latency_target) else "FAIL",
                        "mean_status": "PASS" if a_mean <= mean_latency_target else "FAIL",
                        "p95_status": "PASS" if a_p95 <= p95_latency_target else "FAIL"
                    })
        
        result = {
            "metadata": {
                "time_range": time_range,
                "agent_name": agent_name
            },
            "compliance": compliance,
            "per_agent_compliance": per_agent_compliance,
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
    logging.info(f"Tool Call: save_analysis_report(filename='{filename}')")
    try:
        print(f"DEBUG: save_analysis_report CALLED with filename={filename}") # DEBUG PRINT
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
    logging.info("Tool Call: get_analysis_config()")
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

def analyze_thinking_overhead(
    time_range: str = "24h",
    model_name: Optional[str] = None,
    agent_name: Optional[str] = None
) -> str:
    """Analyze 'thinking' feature overhead by examining thought token patterns.
    
    Identifies queries where thinking dominates the response:
    - High thought/output ratio (>5:1 indicates heavy thinking)
    - Unexpectedly high thought tokens for simple queries
    - Correlation between thought tokens and latency
    
    Returns:
        JSON with thinking patterns, ratios, and recommendations
    """
    logging.info(f"Tool Call: analyze_thinking_overhead(time_range='{time_range}', model_name='{model_name}', agent_name='{agent_name}')")
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
        
        tables = get_table_list()
        
        query = f"""
        SELECT
          CAST(T.request_id AS STRING) AS request_id,
          COALESCE(JSON_VALUE(T.full_request.labels.adk_agent_name), 'unknown') AS agent_name,
          CAST(JSON_EXTRACT_SCALAR(T.metadata, '$.request_latency') AS FLOAT64) / 1000 AS latency,
          SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.promptTokenCount) AS INT64) AS input_tokens,
          SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.candidatesTokenCount) AS INT64) AS output_tokens,
          SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.totalTokenCount) AS INT64) AS total_tokens
        FROM
          `{PROJECT_ID}.{DATASET_ID}.{tables[0]}` AS T
        WHERE
          {where_clause}
        ORDER BY latency DESC
        LIMIT 1000
        """
        
        df = execute_bigquery(query)
        
        if df.empty:
            return json.dumps({"error": "No data found"})
            
        # Calculate thought tokens and ratios
        # Assuming thought tokens are part of total but not explicit in standardized usageMetadata in some versions,
        # or we can infer them if total > input + output. 
        # Note: Some models expose distinct thought tokens, others bundle them.
        # We'll use max(0, total - input - output) as a proxy for overhead/thought if explicit field missing everywhere.
        # But actually, Gemini 2.0 Flash thinking model has specific fields. 
        # For now, we'll use the inference method which aligns with our previous tool updates.
        
        df['input_tokens'] = pd.to_numeric(df['input_tokens'], errors='coerce').fillna(0)
        df['output_tokens'] = pd.to_numeric(df['output_tokens'], errors='coerce').fillna(0)
        df['total_tokens'] = pd.to_numeric(df['total_tokens'], errors='coerce').fillna(0)
        df['thought_tokens'] = (df['total_tokens'] - df['input_tokens'] - df['output_tokens']).clip(lower=0)
        df['thought_output_ratio'] = df.apply(
            lambda row: row['thought_tokens'] / row['output_tokens'] if row['output_tokens'] > 0 else 0, axis=1
        )
        
        # Identify heavy thinking queries
        heavy_thinking = df[df['thought_output_ratio'] > 5]
        
        # Calculate correlations
        metrics = df[['latency', 'thought_tokens', 'output_tokens', 'thought_output_ratio']].corr()['latency']
        
        result = {
            "metadata": {
                "time_range": f"{start_time} to {end_time}",
                "total_analyzed": len(df)
            },
            "statistics": {
                "avg_thought_tokens": float(df['thought_tokens'].mean()),
                "max_thought_tokens": int(df['thought_tokens'].max()),
                "avg_thought_output_ratio": float(df['thought_output_ratio'].mean()),
                "percent_heavy_thinking": float(len(heavy_thinking) / len(df) * 100)
            },
            "correlations": {
                "latency_vs_thought_tokens": float(metrics['thought_tokens']),
                "latency_vs_ratio": float(metrics['thought_output_ratio'])
            },
            "heavy_thinking_samples": heavy_thinking.head(5)[['request_id', 'agent_name', 'latency', 'thought_tokens', 'output_tokens', 'thought_output_ratio']].to_dict('records')
        }
        
        return json.dumps(result, cls=AnalysisEncoder)
        
    except Exception as e:
        error_msg = f"Error in analyze_thinking_overhead: {str(e)}"
        logging.error(error_msg)
        return json.dumps({"error": error_msg})


def detect_compute_inefficiency(
    time_range: str = "24h",
    model_name: Optional[str] = None,
    agent_name: Optional[str] = None
) -> str:
    """Compare actual latency to expected latency to detect compute bottlenecks.
    
    Expected latency model:
    - Prefill: 0.0005s per input token
    - Decode: 0.05s per output token (baseline 20 t/s)
    - Fixed overhead: 0.5s
    
    Flags queries where Actual > 5x Expected.
    """
    logging.info(f"Tool Call: detect_compute_inefficiency(time_range='{time_range}', model_name='{model_name}', agent_name='{agent_name}')")
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
        
        tables = get_table_list()
        
        query = f"""
        SELECT
          CAST(T.request_id AS STRING) AS request_id,
          COALESCE(JSON_VALUE(T.full_request.labels.adk_agent_name), 'unknown') AS agent_name,
          CAST(JSON_EXTRACT_SCALAR(T.metadata, '$.request_latency') AS FLOAT64) / 1000 AS latency,
          SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.promptTokenCount) AS INT64) AS input_tokens,
          SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.candidatesTokenCount) AS INT64) AS output_tokens,
          -- Preview for context
          SUBSTR(TO_JSON_STRING(T.full_request), 1, 100) AS request_preview
        FROM
          `{PROJECT_ID}.{DATASET_ID}.{tables[0]}` AS T
        WHERE
          {where_clause}
        ORDER BY latency DESC
        LIMIT 1000
        """
        
        df = execute_bigquery(query)
        
        if df.empty:
            return json.dumps({"error": "No data found"})
            
        # Calculate expected latency
        # Model: 0.5s overhead + 0.5ms/input + 50ms/output
        df['expected_latency'] = 0.5 + (df['input_tokens'] * 0.0005) + (df['output_tokens'] * 0.05)
        
        # Calculate inefficiency ratio
        df['inefficiency_ratio'] = df['latency'] / df['expected_latency']
        
        # Flag inefficient queries (Actual > 5x Expected)
        inefficient_queries = df[df['inefficiency_ratio'] > 5.0].copy()
        inefficient_queries.sort_values('inefficiency_ratio', ascending=False, inplace=True)
        
        result = {
            "metadata": {
                "time_range": f"{start_time} to {end_time}",
                "total_analyzed": len(df),
                "inefficiency_threshold": "5x expected"
            },
            "summary": {
                "inefficient_count": len(inefficient_queries),
                "inefficient_percentage": float(len(inefficient_queries) / len(df) * 100),
                "avg_inefficiency_ratio": float(inefficient_queries['inefficiency_ratio'].mean()) if not inefficient_queries.empty else 0
            },
            "top_inefficient_queries": inefficient_queries.head(10)[['request_id', 'agent_name', 'latency', 'expected_latency', 'inefficiency_ratio']].to_dict('records')
        }
        
        return json.dumps(result, cls=AnalysisEncoder)
        
    except Exception as e:
        error_msg = f"Error in detect_compute_inefficiency: {str(e)}"
        logging.error(error_msg)
        return json.dumps({"error": error_msg})

def get_generation_config_comparison(
    time_range: str = "24h",
    agent_name: Optional[str] = None,
    model_name: Optional[str] = None
) -> str:
    """
    Compare latency performance across different generation config settings.
    
    Groups requests by temperature and maxOutputTokens to identify which
    configurations lead to better or worse latency performance.
    
    Args:
        time_range: Time range to analyze (default: "24h")
        agent_name: Filter by specific agent (optional)
        model_name: Filter by specific model (optional)
        
    Returns:
        JSON string with per-config performance comparison
    """
    logging.info(f"Tool Call: get_generation_config_comparison(time_range='{time_range}', agent_name='{agent_name}', model_name='{model_name}')")
    try:
        time_range_dict = json.loads(parse_time_range(time_range))
        start_time, end_time = time_range_dict['start_date'], time_range_dict['end_date']
        
        where_clauses = [
            f"T.logging_time BETWEEN '{start_time}' AND '{end_time}'",
            "T.full_request IS NOT NULL",
            "T.full_response IS NOT NULL",
            "JSON_VALUE(T.metadata.request_latency) IS NOT NULL",
            "JSON_VALUE(T.full_request, '$.generationConfig.temperature') IS NOT NULL"
        ]
        
        if agent_name:
            where_clauses.append(f"JSON_VALUE(T.full_request.labels.adk_agent_name) = '{agent_name}'")
        if model_name:
            where_clauses.append(f"T.model LIKE '%{model_name}%'")
        
        where_clause = " AND ".join(where_clauses)
        
        tables = get_table_list()
        
        if len(tables) == 1:
            query = f"""
        SELECT
          CASE 
            WHEN SAFE_CAST(JSON_VALUE(T.full_request, '$.generationConfig.temperature') AS FLOAT64) IS NULL THEN 'Unknown'
            WHEN SAFE_CAST(JSON_VALUE(T.full_request, '$.generationConfig.temperature') AS FLOAT64) <= 0.3 THEN 'Low (0.0-0.3)'
            WHEN SAFE_CAST(JSON_VALUE(T.full_request, '$.generationConfig.temperature') AS FLOAT64) <= 0.7 THEN 'Medium (0.4-0.7)'
            ELSE 'High (0.8-1.0)'
          END AS temperature_range,
          CASE
            WHEN SAFE_CAST(JSON_VALUE(T.full_request, '$.generationConfig.maxOutputTokens') AS INT64) IS NULL THEN 'Unspecified'
            WHEN SAFE_CAST(JSON_VALUE(T.full_request, '$.generationConfig.maxOutputTokens') AS INT64) <= 2048 THEN '≤2048'
            WHEN SAFE_CAST(JSON_VALUE(T.full_request, '$.generationConfig.maxOutputTokens') AS INT64) <= 4096 THEN '2049-4096'
            WHEN SAFE_CAST(JSON_VALUE(T.full_request, '$.generationConfig.maxOutputTokens') AS INT64) <= 8192 THEN '4097-8192'
            ELSE '>8192'
          END AS max_tokens_range,
          COUNT(*) as request_count,
          AVG(CAST(JSON_EXTRACT_SCALAR(T.metadata, '$.request_latency') AS FLOAT64) / 1000) AS avg_latency,
          APPROX_QUANTILES(CAST(JSON_EXTRACT_SCALAR(T.metadata, '$.request_latency') AS FLOAT64) / 1000, 100)[OFFSET(95)] AS p95_latency,
          AVG(SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.candidatesTokenCount) AS INT64)) AS avg_output_tokens,
          AVG(SAFE_CAST(JSON_VALUE(T.full_request, '$.generationConfig.maxOutputTokens') AS INT64)) AS avg_max_tokens_config,
          AVG(SAFE_CAST(JSON_VALUE(T.full_request, '$.generationConfig.temperature') AS FLOAT64)) AS avg_temperature,
          AVG(
            SAFE_DIVIDE(
              SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.candidatesTokenCount) AS INT64),
              SAFE_CAST(JSON_VALUE(T.full_request, '$.generationConfig.maxOutputTokens') AS INT64)
            )
          ) AS token_efficiency_ratio
        FROM
          `{PROJECT_ID}.{DATASET_ID}.{tables[0]}` AS T
        WHERE
          {where_clause}
        GROUP BY temperature_range, max_tokens_range
        ORDER BY request_count DESC
        """
        else:
            # Multiple tables - union approach
            table_selects = []
            for table in tables:
                table_select = f"""
            SELECT
              CASE 
                WHEN SAFE_CAST(JSON_VALUE(T.full_request, '$.generationConfig.temperature') AS FLOAT64) IS NULL THEN 'Unknown'
                WHEN SAFE_CAST(JSON_VALUE(T.full_request, '$.generationConfig.temperature') AS FLOAT64) <= 0.3 THEN 'Low (0.0-0.3)'
                WHEN SAFE_CAST(JSON_VALUE(T.full_request, '$.generationConfig.temperature') AS FLOAT64) <= 0.7 THEN 'Medium (0.4-0.7)'
                ELSE 'High (0.8-1.0)'
              END AS temperature_range,
              CASE
                WHEN SAFE_CAST(JSON_VALUE(T.full_request, '$.generationConfig.maxOutputTokens') AS INT64) IS NULL THEN 'Unspecified'
                WHEN SAFE_CAST(JSON_VALUE(T.full_request, '$.generationConfig.maxOutputTokens') AS INT64) <= 2048 THEN '≤2048'
                WHEN SAFE_CAST(JSON_VALUE(T.full_request, '$.generationConfig.maxOutputTokens') AS INT64) <= 4096 THEN '2049-4096'
                WHEN SAFE_CAST(JSON_VALUE(T.full_request, '$.generationConfig.maxOutputTokens') AS INT64) <= 8192 THEN '4097-8192'
                ELSE '>8192'
              END AS max_tokens_range,
              CAST(JSON_EXTRACT_SCALAR(T.metadata, '$.request_latency') AS FLOAT64) / 1000 AS latency,
              SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.candidatesTokenCount) AS INT64) AS output_tokens,
              SAFE_CAST(JSON_VALUE(T.full_request, '$.generationConfig.maxOutputTokens') AS INT64) AS max_tokens_config,
              SAFE_CAST(JSON_VALUE(T.full_request, '$.generationConfig.temperature') AS FLOAT64) AS temperature,
              SAFE_DIVIDE(
                SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.candidatesTokenCount) AS INT64),
                SAFE_CAST(JSON_VALUE(T.full_request, '$.generationConfig.maxOutputTokens') AS INT64)
              ) AS token_efficiency
            FROM
              `{PROJECT_ID}.{DATASET_ID}.{table}` AS T
            WHERE
              {where_clause}
            """
                table_selects.append(table_select)
            
            union_data = "\nUNION ALL\n".join(table_selects)
            
            query = f"""
        SELECT
          temperature_range,
          max_tokens_range,
          COUNT(*) as request_count,
          AVG(latency) AS avg_latency,
          APPROX_QUANTILES(latency, 100)[OFFSET(95)] AS p95_latency,
          AVG(output_tokens) AS avg_output_tokens,
          AVG(max_tokens_config) AS avg_max_tokens_config,
          AVG(temperature) AS avg_temperature,
          AVG(token_efficiency) AS token_efficiency_ratio
        FROM (
{union_data}
        )
        GROUP BY temperature_range, max_tokens_range
        ORDER BY request_count DESC
        """
        
        df = execute_bigquery(query)
        
        if df.empty:
            return json.dumps({
                "error": "No data found with generationConfig information",
                "metadata": {"time_range": f"{start_time} to {end_time}"}
            })
        
        configs = []
        for _, row in df.iterrows():
            config_entry = {
                "temperature_range": row['temperature_range'],
                "max_tokens_range": row['max_tokens_range'],
                "request_count": int(row['request_count']),
                "avg_latency": float(row['avg_latency']) if pd.notna(row['avg_latency']) else None,
                "p95_latency": float(row['p95_latency']) if pd.notna(row['p95_latency']) else None,
                "avg_output_tokens": float(row['avg_output_tokens']) if pd.notna(row['avg_output_tokens']) else None,
                "avg_max_tokens_config": float(row['avg_max_tokens_config']) if pd.notna(row['avg_max_tokens_config']) else None,
                "avg_temperature": float(row['avg_temperature']) if pd.notna(row['avg_temperature']) else None,
                "token_efficiency_ratio": float(row['token_efficiency_ratio']) if pd.notna(row['token_efficiency_ratio']) else None
            }
            configs.append(config_entry)
        
        # Find best and worst configs
        valid_configs = [c for c in configs if c['avg_latency'] is not None]
        if valid_configs:
            best_config = min(valid_configs, key=lambda x: x['avg_latency'])
            worst_config = max(valid_configs, key=lambda x: x['avg_latency'])
            
            # Find most efficient (highest token efficiency ratio)
            efficient_configs = [c for c in configs if c['token_efficiency_ratio'] is not None]
            most_efficient = max(efficient_configs, key=lambda x: x['token_efficiency_ratio']) if efficient_configs else None
        else:
            best_config = None
            worst_config = None
            most_efficient = None
        
        result = {
            "metadata": {
                "time_range": f"{start_time} to {end_time}",
                "agent_name": agent_name or "all",
                "model_name": model_name or "all",
                "total_requests": int(df['request_count'].sum())
            },
            "configurations": configs,
            "insights": {
                "best_latency_config": {
                    "temperature": best_config['temperature_range'],
                    "max_tokens": best_config['max_tokens_range'],
                    "avg_latency": best_config['avg_latency']
                } if best_config else None,
                "worst_latency_config": {
                    "temperature": worst_config['temperature_range'],
                    "max_tokens": worst_config['max_tokens_range'],
                    "avg_latency": worst_config['avg_latency']
                } if worst_config else None,
                "most_efficient_config": {
                    "temperature": most_efficient['temperature_range'],
                    "max_tokens": most_efficient['max_tokens_range'],
                    "efficiency_ratio": most_efficient['token_efficiency_ratio']
                } if most_efficient else None
            },
            "summary": f"Analyzed {len(configs)} config combinations. Best latency: {best_config['temperature_range']} temp, {best_config['max_tokens_range']} tokens ({best_config['avg_latency']:.2f}s)" if best_config else "No data"
        }
        
        return json.dumps(result, cls=AnalysisEncoder)
        
    except Exception as e:
        error_msg = f"Error in get_generation_config_comparison: {str(e)}"
        logging.error(error_msg)
        return json.dumps({"error": error_msg})


def analyze_config_correlation(
    time_range: str = "24h",
    agent_name: Optional[str] = None,
    model_name: Optional[str] = None
) -> str:
    """
    Analyze correlation between generation config parameters and latency.
    
    Calculates correlation coefficients for:
    - Temperature vs latency
    - maxOutputTokens vs latency
    - topK vs latency (if available)
    - topP vs latency (if available)
    
    Args:
        time_range: Time range to analyze (default: "24h")
        agent_name: Filter by specific agent (optional)
        model_name: Filter by specific model (optional)
        
    Returns:
        JSON string with correlation analysis and scatter plot data
    """
    logging.info(f"Tool Call: analyze_config_correlation(time_range='{time_range}', agent_name='{agent_name}', model_name='{model_name}')")
    try:
        time_range_dict = json.loads(parse_time_range(time_range))
        start_time, end_time = time_range_dict['start_date'], time_range_dict['end_date']
        
        where_clauses = [
            f"T.logging_time BETWEEN '{start_time}' AND '{end_time}'",
            "T.full_request IS NOT NULL",
            "T.full_response IS NOT NULL",
            "JSON_VALUE(T.metadata.request_latency) IS NOT NULL"
        ]
        
        if agent_name:
            where_clauses.append(f"JSON_VALUE(T.full_request.labels.adk_agent_name) = '{agent_name}'")
        if model_name:
            where_clauses.append(f"T.model LIKE '%{model_name}%'")
        
        where_clause = " AND ".join(where_clauses)
        
        tables = get_table_list()
        
        # Build query for all tables
        if len(tables) == 1:
            tables = get_table_list()
            
            query = f"""
        SELECT
          CAST(JSON_EXTRACT_SCALAR(T.metadata, '$.request_latency') AS FLOAT64) / 1000 AS latency,
          SAFE_CAST(JSON_VALUE(T.full_request, '$.generationConfig.temperature') AS FLOAT64) AS temperature,
          SAFE_CAST(JSON_VALUE(T.full_request, '$.generationConfig.maxOutputTokens') AS INT64) AS max_output_tokens,
          SAFE_CAST(JSON_VALUE(T.full_request, '$.generationConfig.topK') AS INT64) AS top_k,
          SAFE_CAST(JSON_VALUE(T.full_request, '$.generationConfig.topP') AS FLOAT64) AS top_p,
          SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.candidatesTokenCount) AS INT64) AS output_tokens
        FROM
          `{PROJECT_ID}.{DATASET_ID}.{tables[0]}` AS T
        WHERE
          {where_clause}
        """
        else:
            table_selects = []
            for table in tables:
                table_select = f"""
            SELECT
              CAST(JSON_EXTRACT_SCALAR(T.metadata, '$.request_latency') AS FLOAT64) / 1000 AS latency,
              SAFE_CAST(JSON_VALUE(T.full_request, '$.generationConfig.temperature') AS FLOAT64) AS temperature,
              SAFE_CAST(JSON_VALUE(T.full_request, '$.generationConfig.maxOutputTokens') AS INT64) AS max_output_tokens,
              SAFE_CAST(JSON_VALUE(T.full_request, '$.generationConfig.topK') AS INT64) AS top_k,
              SAFE_CAST(JSON_VALUE(T.full_request, '$.generationConfig.topP') AS FLOAT64) AS top_p,
              SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.candidatesTokenCount) AS INT64) AS output_tokens
            FROM
              `{PROJECT_ID}.{DATASET_ID}.{table}` AS T
            WHERE
              {where_clause}
            """
                table_selects.append(table_select)
            
            query = "\nUNION ALL\n".join(table_selects)
        
        df = execute_bigquery(query)
        
        if df.empty or len(df) < 10:
            return json.dumps({
                "error": "Insufficient data for correlation analysis",
                "metadata": {"time_range": f"{start_time} to {end_time}", "rows": len(df)}
            })
        
        # Calculate correlations
        correlations = {}
        scatter_data = {}
        
        # Temperature vs Latency
        temp_data = df[['latency', 'temperature']].dropna()
        if len(temp_data) >= 10:
            corr = temp_data['latency'].corr(temp_data['temperature'])
            correlations['temperature_vs_latency'] = {
                "correlation": float(corr),
                "sample_size": len(temp_data),
                "interpretation": "strong" if abs(corr) > 0.7 else "moderate" if abs(corr) > 0.4 else "weak"
            }
            # Sample data for scatter plot
            sample = temp_data.sample(n=min(100, len(temp_data)))
            scatter_data['temperature_vs_latency'] = sample.to_dict('records')
        
        # MaxOutputTokens vs Latency
        max_tokens_data = df[['latency', 'max_output_tokens']].dropna()
        if len(max_tokens_data) >= 10:
            corr = max_tokens_data['latency'].corr(max_tokens_data['max_output_tokens'])
            correlations['max_output_tokens_vs_latency'] = {
                "correlation": float(corr),
                "sample_size": len(max_tokens_data),
                "interpretation": "strong" if abs(corr) > 0.7 else "moderate" if abs(corr) > 0.4 else "weak"
            }
            sample = max_tokens_data.sample(n=min(100, len(max_tokens_data)))
            scatter_data['max_output_tokens_vs_latency'] = sample.to_dict('records')
        
        # TopK vs Latency
        topk_data = df[['latency', 'top_k']].dropna()
        if len(topk_data) >= 10:
            corr = topk_data['latency'].corr(topk_data['top_k'])
            correlations['top_k_vs_latency'] = {
                "correlation": float(corr),
                "sample_size": len(topk_data),
                "interpretation": "strong" if abs(corr) > 0.7 else "moderate" if abs(corr) > 0.4 else "weak"
            }
        
        # TopP vs Latency
        topp_data = df[['latency', 'top_p']].dropna()
        if len(topp_data) >= 10:
            corr = topp_data['latency'].corr(topp_data['top_p'])
            correlations['top_p_vs_latency'] = {
                "correlation": float(corr),
                "sample_size": len(topp_data),
                "interpretation": "strong" if abs(corr) > 0.7 else "moderate" if abs(corr) > 0.4 else "weak"
            }
        
        # Additional insight: Actual output tokens vs configured max
        output_vs_max_data = df[['output_tokens', 'max_output_tokens']].dropna()
        if len(output_vs_max_data) >= 10:
            avg_utilization = (output_vs_max_data['output_tokens'] / output_vs_max_data['max_output_tokens']).mean()
            correlations['output_vs_max_tokens'] = {
                "avg_utilization": float(avg_utilization),
                "sample_size": len(output_vs_max_data),
                "interpretation": "over-provisioned" if avg_utilization < 0.5 else "well-tuned" if avg_utilization < 0.8 else "near-limit"
            }
        
        result = {
            "metadata": {
                "time_range": f"{start_time} to {end_time}",
                "agent_name": agent_name or "all",
                "model_name": model_name or "all",
                "total_requests": len(df)
            },
            "correlations": correlations,
            "scatter_data": scatter_data,
            "summary": f"Analyzed {len(correlations)} config parameters. Temperature correlation: {correlations.get('temperature_vs_latency', {}).get('correlation', 'N/A')}"
        }
        
        return json.dumps(result, cls=AnalysisEncoder)
        
    except Exception as e:
        error_msg = f"Error in analyze_config_correlation: {str(e)}"
        logging.error(error_msg)
        return json.dumps({"error": error_msg})


def get_config_outliers(
    time_range: str = "24h",
    threshold_efficiency: float = 0.3
) -> str:
    """
    Identify requests with suboptimal generation config settings.
    
    Detects:
    - Requests with maxOutputTokens much higher than actual output (wasteful)
    - Queries with potentially inappropriate temperature settings
    - Over-provisioned configurations
    
    Args:
        time_range: Time range to analyze (default: "24h")
        threshold_efficiency: Efficiency threshold (output/max_output) below which to flag (default: 0.3)
        
    Returns:
        JSON string with config outliers and optimization recommendations
    """
    logging.info(f"Tool Call: get_config_outliers(time_range='{time_range}', threshold_efficiency={threshold_efficiency})")
    try:
        time_range_dict = json.loads(parse_time_range(time_range))
        start_time, end_time = time_range_dict['start_date'], time_range_dict['end_date']
        
        where_clauses = [
            f"T.logging_time BETWEEN '{start_time}' AND '{end_time}'",
            "T.full_request IS NOT NULL",
            "T.full_response IS NOT NULL",
            "JSON_VALUE(T.metadata.request_latency) IS NOT NULL",
            "JSON_VALUE(T.full_request, '$.generationConfig.maxOutputTokens') IS NOT NULL"
        ]
        
        where_clause = " AND ".join(where_clauses)
        
        tables = get_table_list()
        
        if len(tables) == 1:
            query = f"""
        SELECT
          T.request_id,
          COALESCE(JSON_VALUE(T.full_request.labels.adk_agent_name), 'unknown') AS agent_name,
          CAST(JSON_EXTRACT_SCALAR(T.metadata, '$.request_latency') AS FLOAT64) / 1000 AS latency,
          SAFE_CAST(JSON_VALUE(T.full_request, '$.generationConfig.temperature') AS FLOAT64) AS temperature,
          SAFE_CAST(JSON_VALUE(T.full_request, '$.generationConfig.maxOutputTokens') AS INT64) AS max_output_tokens,
          SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.candidatesTokenCount) AS INT64) AS output_tokens,
          SAFE_DIVIDE(
            SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.candidatesTokenCount) AS INT64),
            SAFE_CAST(JSON_VALUE(T.full_request, '$.generationConfig.maxOutputTokens') AS INT64)
          ) AS token_efficiency
        FROM
          `{PROJECT_ID}.{DATASET_ID}.{tables[0]}` AS T
        WHERE
          {where_clause}
          AND SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.candidatesTokenCount) AS INT64) IS NOT NULL
        """
        else:
            table_selects = []
            for table in tables:
                table_select = f"""
            SELECT
              T.request_id,
              COALESCE(JSON_VALUE(T.full_request.labels.adk_agent_name), 'unknown') AS agent_name,
              CAST(JSON_EXTRACT_SCALAR(T.metadata, '$.request_latency') AS FLOAT64) / 1000 AS latency,
              SAFE_CAST(JSON_VALUE(T.full_request, '$.generationConfig.temperature') AS FLOAT64) AS temperature,
              SAFE_CAST(JSON_VALUE(T.full_request, '$.generationConfig.maxOutputTokens') AS INT64) AS max_output_tokens,
              SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.candidatesTokenCount) AS INT64) AS output_tokens,
              SAFE_DIVIDE(
                SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.candidatesTokenCount) AS INT64),
                SAFE_CAST(JSON_VALUE(T.full_request, '$.generationConfig.maxOutputTokens') AS INT64)
              ) AS token_efficiency
            FROM
              `{PROJECT_ID}.{DATASET_ID}.{table}` AS T
            WHERE
              {where_clause}
              AND SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.candidatesTokenCount) AS INT64) IS NOT NULL
            """
                table_selects.append(table_select)
            
            query = "\nUNION ALL\n".join(table_selects)
        
        df = execute_bigquery(query)
        
        if df.empty:
            return json.dumps({
                "error": "No data found with config information",
                "metadata": {"time_range": f"{start_time} to {end_time}"}
            })
        
        # Find outliers: low token efficiency (over-provisioned maxOutputTokens)
        wasteful_configs = df[df['token_efficiency'] < threshold_efficiency].copy()
        wasteful_configs['waste_tokens'] = wasteful_configs['max_output_tokens'] - wasteful_configs['output_tokens']
        wasteful_configs.sort_values('waste_tokens', ascending=False, inplace=True)
        
        # Calculate potential savings
        total_waste = wasteful_configs['waste_tokens'].sum()
        avg_efficiency = df['token_efficiency'].mean()
        
        # Group by agent to find patterns
        agent_efficiency = df.groupby('agent_name').agg({
            'token_efficiency': 'mean',
            'max_output_tokens': 'mean',
            'output_tokens': 'mean'
        }).reset_index()
        agent_efficiency['recommended_max_tokens'] = (agent_efficiency['output_tokens'] * 1.5).round(0)  # 50% buffer
        
        outliers_list = []
        for _, row in wasteful_configs.head(20).iterrows():
            outlier = {
                "request_id": row['request_id'],
                "agent_name": row['agent_name'],
                "latency": float(row['latency']),
                "temperature": float(row['temperature']) if pd.notna(row['temperature']) else None,
                "max_output_tokens_config": int(row['max_output_tokens']),
                "actual_output_tokens": int(row['output_tokens']),
                "token_efficiency": float(row['token_efficiency']),
                "wasted_tokens": int(row['waste_tokens']),
                "recommendation": f"Reduce maxOutputTokens from {int(row['max_output_tokens'])} to ~{int(row['output_tokens'] * 1.5)}"
            }
            outliers_list.append(outlier)
        
        agent_recommendations = []
        for _, row in agent_efficiency.iterrows():
            if row['token_efficiency'] < threshold_efficiency:
                agent_rec = {
                    "agent_name": row['agent_name'],
                    "current_avg_max_tokens": int(row['max_output_tokens']),
                    "actual_avg_output": int(row['output_tokens']),
                    "efficiency": float(row['token_efficiency']),
                    "recommended_max_tokens": int(row['recommended_max_tokens'])
                }
                agent_recommendations.append(agent_rec)
        
        result = {
            "metadata": {
                "time_range": f"{start_time} to {end_time}",
                "total_requests": len(df),
                "threshold_efficiency": threshold_efficiency
            },
            "summary": {
                "wasteful_configs_count": len(wasteful_configs),
                "wasteful_percentage": float(len(wasteful_configs) / len(df) * 100),
                "total_wasted_tokens": int(total_waste),
                "avg_token_efficiency": float(avg_efficiency)
            },
            "outliers": outliers_list,
            "agent_recommendations": agent_recommendations,
            "overall_recommendation": f"Found {len(wasteful_configs)} requests ({len(wasteful_configs)/len(df)*100:.1f}%) with <{threshold_efficiency*100}% token efficiency. Consider reducing maxOutputTokens for affected agents."
        }
        
        return json.dumps(result, cls=AnalysisEncoder)
        
    except Exception as e:
        error_msg = f"Error in get_config_outliers: {str(e)}"
        logging.error(error_msg)
        return json.dumps({"error": error_msg})


# =========================================
# LATENCY DIMENSIONS (New Architecture)
# =========================================
CURRENT_DIMENSION_LIST = [
    "KPI Compliance & Overall Statistics",
    "Hourly & Daily Patterns",
    "Token Usage & Correlation",
    "Micro-Burst & Queuing Analysis",
    "Model & Agent Performance Comparison",
    "Slow Query Deep Dive",
    "Cost & Efficiency Analysis"
]

def set_dimensions_and_transfer(tool_context: ToolContext, dimensions: list[str]) -> dict:
    """Sets the DIMENSIONS_LIST in session state and transfers control to the processing loop."""
    logging.info(f"Tool Call: set_dimensions_and_transfer(dimensions={dimensions})")
    logging.info(f"Setting DIMENSIONS_LIST to: {dimensions}")
    tool_context.state["DIMENSIONS_LIST"] = dimensions
    tool_context.state["DIMENSION_INDEX"] = 0
    tool_context.state["LAST_ACTION"] = "Sequential"
    tool_context.actions.transfer_to_agent = "dimension_processing_loop"
    return {"status": "success", "message": f"Prepared to process dimensions: {dimensions}."}

def trigger_latency_parallel_report(tool_context: ToolContext) -> dict:
    """Transfers control to the complete_report_generator for parallel processing."""
    logging.info("Tool Call: trigger_latency_parallel_report()")
    logging.info("Triggering complete_report_generator for parallel processing.")
    tool_context.state["LAST_ACTION"] = "Parallel"
    tool_context.actions.transfer_to_agent = "complete_report_generator"
    return {"status": "success", "message": "Starting complete report generation in parallel."}

def process_latency_question(tool_context: ToolContext, dimension_name: str, user_question: str) -> dict:
    """Sets up the session state to process a single user-provided question within a specific dimension."""
    logging.info(f"Tool Call: process_latency_question(dimension_name='{dimension_name}', user_question='{user_question}')")
    logging.info(f"Processing user question for dimension '{dimension_name}': '{user_question}'")
    tool_context.state["CURRENT_DIMENSION"] = dimension_name
    tool_context.state["LAST_ACTION"] = "Question"
    tool_context.state["STRATEGIST_OUTPUT"] = f"* {user_question}"
    tool_context.state["INVESTIGATION_FEEDBACK"] = None
    tool_context.actions.transfer_to_agent = "latency_analysis_team"
    return {"status": "success", "message": f"Investigating question in '{dimension_name}': '{user_question}'"}

def read_report_content(tool_context: ToolContext) -> str:
    """Reads the current report content from session state (Parallel or Sequential)."""
    logging.info("Tool Call: read_report_content()")
    if "FINAL_REPORT_MARKDOWN" in tool_context.state:
        return tool_context.state["FINAL_REPORT_MARKDOWN"]
    if "CURRENT_REPORT_MARKDOWN" in tool_context.state:
        return tool_context.state["CURRENT_REPORT_MARKDOWN"]
    return ""

def has_report_content(tool_context: ToolContext) -> bool:
    """Checks if there is any generated report content in the session state."""
    return "FINAL_REPORT_MARKDOWN" in tool_context.state or "CURRENT_REPORT_MARKDOWN" in tool_context.state

async def accumulate_investigator_output(
    callback_context: CallbackContext,
    llm_response: LlmResponse
) -> Optional[LlmResponse]:
    """
    Accumulates investigator output.
    """
    agent_name = callback_context.agent_name
    logging.info(f"[{agent_name}] --- In accumulate_investigator_output ---")
    return None

# Alias for agent convenience
get_request_details = get_query_details

def get_daily_patterns(
    time_range: str = "24h",
    model_name: Optional[str] = None,
    agent_name: Optional[str] = None
) -> str:
    """
    Get daily latency patterns including P95 and Day-of-Week breakdown.
    
    Returns:
        JSON string with daily statistics (request count, p50, p95, avg latency)
        grouped by Day of Week (1=Sunday, 2=Monday, ..., 7=Saturday).
    """
    logging.info(f"Tool Call: get_daily_patterns(time_range='{time_range}', model_name='{model_name}', agent_name='{agent_name}')")
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
        
        # Use simple union source provided by helper
        union_source = build_multi_table_source(where_clause, select_suffix="AS T")
        
        query = f"""
        SELECT
          EXTRACT(DAYOFWEEK FROM T.logging_time) AS day_num,
          CASE EXTRACT(DAYOFWEEK FROM T.logging_time)
            WHEN 1 THEN 'Sunday'
            WHEN 2 THEN 'Monday'
            WHEN 3 THEN 'Tuesday'
            WHEN 4 THEN 'Wednesday'
            WHEN 5 THEN 'Thursday'
            WHEN 6 THEN 'Friday'
            WHEN 7 THEN 'Saturday'
          END AS day_name,
          COUNT(*) as request_count,
          AVG(CAST(JSON_EXTRACT_SCALAR(T.metadata, '$.request_latency') AS FLOAT64) / 1000) AS avg_latency,
          APPROX_QUANTILES(CAST(JSON_EXTRACT_SCALAR(T.metadata, '$.request_latency') AS FLOAT64) / 1000, 100)[OFFSET(50)] AS p50_latency,
          APPROX_QUANTILES(CAST(JSON_EXTRACT_SCALAR(T.metadata, '$.request_latency') AS FLOAT64) / 1000, 100)[OFFSET(95)] AS p95_latency
        FROM (
            {union_source}
        )
        GROUP BY day_num, day_name
        ORDER BY day_num
        """
        
        df = execute_bigquery(query)
        
        if df.empty:
            return json.dumps({"error": "No data found for the given time range."})
        
        results = []
        for _, row in df.iterrows():
            results.append({
                "day": row['day_name'],
                "request_count": int(row['request_count']),
                "avg_latency": round(float(row['avg_latency']), 2),
                "p50_latency": round(float(row['p50_latency']), 2),
                "p95_latency": round(float(row['p95_latency']), 2)
            })
            
        return json.dumps({
            "metadata": {
                "time_range": f"{start_time} to {end_time}",
                "metric": "Daily Latency Patterns"
            },
            "daily_stats": results
        }, indent=2)
        
    except Exception as e:
        error_msg = f"Error in get_daily_patterns: {str(e)}"
        logging.error(error_msg)
        return json.dumps({"error": error_msg})


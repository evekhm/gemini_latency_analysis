# utils.py
import os
import json
import logging
from decimal import Decimal
from google.cloud import bigquery
from dotenv import load_dotenv

load_dotenv()

PROJECT_ID = os.getenv('PROJECT_ID')
DATASET = os.getenv('DATASET', 'gemini_logs') # Default to gemini_logs if not set, but usually in .env
GEMINI_LOG_TABLE = os.getenv('GEMINI_LOG_TABLE', 'gemini_logs')

# Custom JSON encoder to handle Decimal types from BigQuery
class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        return super(DecimalEncoder, self).default(obj)

def fetch_slow_queries(num_records: int = 10) -> str:
    """
    Fetches the top N slowest queries from the BigQuery logs and returns metadata only.
    
    Args:
        num_records: The number of records to fetch. Defaults to 10.
        
    Returns:
        A JSON string containing the count and list of request IDs.
    """
    if not PROJECT_ID:
        return json.dumps({"error": "PROJECT_ID environment variable is not set."})
    
    project_id = PROJECT_ID
    dataset = DATASET
    table_id = GEMINI_LOG_TABLE

    # Initialize BigQuery client
    client = bigquery.Client(project=project_id)
    
    # Construct the query - fetch only metadata first
    query = f"""
        SELECT
          T.request_id,
          ROUND(SAFE_CAST(JSON_VALUE(T.metadata.request_latency) AS FLOAT64) / 1000.0, 2) AS request_latency_seconds
        FROM
          `{project_id}.{dataset}.{table_id}` AS T
        WHERE
          T.full_request IS NOT NULL
          AND T.full_response IS NOT NULL
        ORDER BY
          request_latency_seconds DESC
        LIMIT {num_records}
    """
    
    try:
        logging.info(f"Executing BigQuery query on {table_id} with limit {num_records}")
        query_job = client.query(query)
        results = query_job.result()
        
        # Convert results to a list of request IDs
        request_ids = []
        for row in results:
            request_ids.append({
                "request_id": str(row["request_id"]),  # Convert to string to preserve full value
                "latency_seconds": float(row["request_latency_seconds"]) if row["request_latency_seconds"] else 0.0
            })
        
        logging.info(f"Successfully fetched {len(request_ids)} request IDs")
        return json.dumps({
            "count": len(request_ids),
            "requests": request_ids
        }, cls=DecimalEncoder)
    
    except Exception as e:
        error_msg = f"Error fetching slow queries: {str(e)}"
        logging.error(error_msg)
        return json.dumps({"error": error_msg})


def fetch_single_query(request_id: str) -> str:
    """
    Fetches a single query's full details by request_id.
    
    Args:
        request_id: The request ID to fetch.
        
    Returns:
        A JSON string containing the full query details.
    """
    if not PROJECT_ID:
        return json.dumps({"error": "PROJECT_ID environment variable is not set."})
    
    project_id = PROJECT_ID
    dataset = DATASET
    table_id = GEMINI_LOG_TABLE

    # Initialize BigQuery client
    client = bigquery.Client(project=project_id)
    
    # Construct the query for a single record - use parameterized query
    query = f"""
        SELECT
          T.logging_time,
          T.request_id,
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
          `{project_id}.{dataset}.{table_id}` AS T
        WHERE
          CAST(T.request_id AS STRING) = @request_id
        LIMIT 1
    """
    
    try:
        logging.info(f"Fetching single query with request_id: {request_id}")
        
        # Use parameterized query to avoid type issues
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("request_id", "STRING", str(request_id))
            ]
        )
        
        query_job = client.query(query, job_config=job_config)
        results = query_job.result()
        
        # Convert result to dictionary
        for row in results:
            record = dict(row)
            logging.info(f"Successfully fetched query {request_id}")
            return json.dumps(record, default=str)
        
        return json.dumps({"error": f"No record found for request_id: {request_id}"})
    
    except Exception as e:
        error_msg = f"Error fetching query {request_id}: {str(e)}"
        logging.error(error_msg)
        return json.dumps({"error": error_msg})

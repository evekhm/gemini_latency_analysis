#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source environment variables from .env file
# Ensure this file contains PROJECT_ID and MODEL
ENV_FILE="${SCRIPT_DIR}/.env"
if [ -f "${ENV_FILE}" ]; then
    source "${ENV_FILE}"
else
    echo "Error: .env file not found at ${ENV_FILE}"
    exit 1
fi


# Verify that the required variables are set
if [ -z "$PROJECT_ID" ]; then
  echo "Error: PROJECT_ID must be set in the .env file."
  exit 1
fi

echo "Querying BigQuery table for model: $MODEL"
echo "Project: $PROJECT_ID"
echo "Dataset: $DATASET"
echo "Table: $GEMINI_LOG_TABLE"
echo "---"

# Construct the fully qualified table name for the query
TABLE_ID="${PROJECT_ID}.${DATASET}.${GEMINI_LOG_TABLE}"

# Run the BigQuery query
# The condition "AND JSON_VALUE(T.full_request.labels.adk_agent_name) IS NOT NULL" has been removed.
bq query --use_legacy_sql=false "
SELECT
  T.logging_time,
  T.request_id,
  JSON_VALUE(T.full_request.labels.adk_agent_name) AS adk_agent_name,
  ROUND(SAFE_CAST(JSON_VALUE(T.metadata.request_latency) AS FLOAT64) / 1000.0, 2) AS request_latency_seconds,
  SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.thoughtsTokenCount) AS INT64) AS thoughts_token_count,
  SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.candidatesTokenCount) AS INT64) AS output_token_count,
  SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.promptTokenCount) AS INT64) AS prompt_token_count,
  SAFE_CAST(JSON_VALUE(T.full_response.usageMetadata.totalTokenCount) AS INT64) AS total_token_count
FROM
  \`${TABLE_ID}\` AS T
WHERE
  T.full_request IS NOT NULL
  AND T.full_response IS NOT NULL
;
"

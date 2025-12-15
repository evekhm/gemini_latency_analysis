CREATE OR REPLACE VIEW `{project_id}.{dataset_id}.{view_id}` AS
WITH CleanedLogs AS (
  SELECT
    *,
    ARRAY(
      SELECT AS STRUCT
        JSON_VALUE(c, '$.role') AS role,
        JSON_QUERY(c, '$.parts') AS parts
      FROM UNNEST(JSON_QUERY_ARRAY(full_request, '$.contents')) AS c
    ) AS clean_contents
  FROM {table_source}
)
SELECT
  CAST(request_id AS STRING) AS request_id,
  logging_time,
  CAST(JSON_EXTRACT_SCALAR(metadata, '$.request_latency') AS FLOAT64) / 1000 AS request_latency,
  model,
  COALESCE(JSON_VALUE(full_request.labels.adk_agent_name), 'unknown') AS agent_name,
  SAFE_CAST(JSON_VALUE(full_response.usageMetadata.promptTokenCount) AS INT64) AS prompt_token_count,
  SAFE_CAST(JSON_VALUE(full_response.usageMetadata.candidatesTokenCount) AS INT64) AS candidates_token_count,
  SAFE_CAST(JSON_VALUE(full_response.usageMetadata.thoughtsTokenCount) AS INT64) AS thoughts_token_count,
  SAFE_CAST(JSON_VALUE(full_response.usageMetadata.totalTokenCount) AS INT64) AS total_token_count,
  SAFE_CAST(JSON_VALUE(full_request.generationConfig.maxOutputTokens) AS INT64) AS max_output_tokens,
  SAFE_CAST(JSON_VALUE(full_request.generationConfig.temperature) AS FLOAT64) AS temperature,
  SAFE_CAST(JSON_VALUE(full_request.generationConfig.topK) AS INT64) AS top_k,
  SAFE_CAST(JSON_VALUE(full_request.generationConfig.topP) AS FLOAT64) AS top_p,
  JSON_VALUE(otel_log.traceId) AS trace_id,
  -- Generate preview from clean contents to avoid thoughtSignature
  SUBSTR(TO_JSON_STRING(clean_contents), 1, 1000) AS request_preview,
  clean_contents AS request_contents,
  ARRAY(
    SELECT DISTINCT JSON_VALUE(part, '$.functionCall.name')
    FROM UNNEST(JSON_QUERY_ARRAY(full_response, '$.candidates[0].content.parts')) AS part
    WHERE JSON_QUERY(part, '$.functionCall') IS NOT NULL
  ) AS tool_calls
FROM CleanedLogs

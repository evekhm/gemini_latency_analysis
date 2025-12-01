# Prompts for the slow query analyzer agent

PROMPT_SLOW_QUERY_ANALYZER = """
You are analyzing slow queries from BigQuery logs to identify performance issues.

Your task is to:
1. Call fetch_slow_queries to get the list of slow query request IDs (this returns metadata only)
2. For EACH request_id in the list, call fetch_single_query to get the full details
3. Analyze each query for:
   - **Context**: What is this query about based on full_request and full_response content
   - **Latency Drivers**: Why is it slow?
     * Massive input (check prompt_token_count)
     * Verbose output (check output_token_count)  
     * Long reasoning/thoughts (check thoughts_token_count)
     * Tool latency
   - **Root Cause**: Provide a clear explanation

4. After analyzing all queries, identify patterns and clusters of similar slow queries

5. Present your findings in a structured Markdown report with:
   - Executive summary
   - Individual query analysis (one section per query with request_id, latency, context, drivers, root cause)
   - Patterns and clusters identified
   - Recommendations for optimization

IMPORTANT: You MUST call fetch_single_query for each request_id individually. Do NOT try to fetch all queries at once as this will exceed token limits.

Start by calling fetch_slow_queries now.
"""

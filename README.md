<!-- TOC -->
* [Latency Analysis](#latency-analysis)
  * [Overview](#overview)
  * [Prerequisites](#prerequisites)
    * [Enable Gemini Requests Logging](#enable-gemini-requests-logging)
    * [Install Libraries](#install-libraries)
  * [Quick start](#quick-start)
    * [Monitor Progress](#monitor-progress)
  * [Load Generator](#load-generator)
    * [Features](#features)
    * [Usage](#usage)
  * [Latency Analyzer Agent](#latency-analyzer-agent)
    * [Features](#features-1)
    * [Analysis Tools](#analysis-tools)
    * [Usage](#usage-1)
  * [Visualization and Charts (Alternative Approach)](#visualization-and-charts-alternative-approach)
    * [Generate Analysis Charts](#generate-analysis-charts)
    * [Generated Charts and Visualizations](#generated-charts-and-visualizations)
      * [1. Agent Summary Analysis](#1-agent-summary-analysis)
      * [2. Latency Distribution Analysis](#2-latency-distribution-analysis)
      * [3. Latency vs Output Tokens](#3-latency-vs-output-tokens)
      * [4. Latency vs Input Tokens](#4-latency-vs-input-tokens)
      * [5. Latency vs Output+Thought Tokens](#5-latency-vs-outputthought-tokens)
      * [6. Hourly Analysis by Day Type](#6-hourly-analysis-by-day-type)
    * [Output Files](#output-files)
  * [Troubleshooting](#troubleshooting)
    * [Debugging Agent Tool Errors](#debugging-agent-tool-errors)
      * [1. Check Agent Logs](#1-check-agent-logs)
      * [2. Test Tools Directly](#2-test-tools-directly)
      * [3. Common Issues and Solutions](#3-common-issues-and-solutions)
      * [4. Enable Verbose Logging](#4-enable-verbose-logging)
      * [5. Verify BigQuery Access](#5-verify-bigquery-access)
      * [6. Check Tool Return Format](#6-check-tool-return-format)
  * [Verification](#verification)
    * [Automated Tests](#automated-tests)
<!-- TOC -->
# Latency Analysis

## Overview

An AI-powered performance analytics platform that automatically analyzes LLM application latency, identifies bottlenecks, and provides actionable optimization recommendations.

**What it does:**
- Analyzes BigQuery logs of LLM calls to identify why queries are slow
- Uses AI-driven hypothesis testing to find root causes (token size, agent design, concurrency, time patterns)
- Clusters similar slow queries to identify systemic issues
- Tracks token usage and estimates costs by agent
- Provides prioritized recommendations with expected impact
- Performs deep-dive analysis of individual slow queries

**Key benefits:**
- Replaces manual chart analysis with automated reports
- Identifies optimization opportunities to reduce token usage
- Pinpoints specific actions to improve latency
- Detects performance degradation trends before they impact users

---

## Prerequisites

### Enable Gemini Requests Logging

Vertex AI can [log samples of requests and responses](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/request-response-logging) for Gemini and supported partner models.
The logs are saved to a BigQuery table for viewing and analysis.

1. Specify PROJECT_ID in [.env](.env):

    ```bash
    export PROJECT_ID="..."
    ```

2. Load environment variables:
    ```shell
    source .env
    ```

3. Set active project:
    ```shell
    gcloud config set project ${PROJECT_ID}
    ```

4. Authenticate::
    ```shell
    gcloud auth login
    gcloud auth application-default login
    ```

5. Enabled required APIs:
    ```shell
    gcloud services enable aiplatform.googleapis.com cloudtrace.googleapis.com --project="${PROJECT_ID}"    
   ```

6. For each LLM Model you want to enable BQ logging for, follow the steps below.

   - **Setup configuration for logging** (see [.env-2.5pro](.env-2.5pro))
     ```bash
       export MODEL_ID="gemini-2.5-pro"  # Gemini Model for which configuration is applied
       export TABLE_ID="2p5-pro"         # name of the table to save logs into
       export REGION="us-central1"  
     ```

   - **Create `request.json` file**
      ```shell
      cat > request.json <<EOF
      {
        "publisherModelConfig": {
           "loggingConfig": {
             "enabled": true,
             "samplingRate": 1,
             "bigqueryDestination": {
               "outputUri": "bq://${PROJECT_ID}.${DATASET_ID}.${TABLE_ID}"
             },
             "enableOtelLogging": true
           }
         }
       }
      EOF
      ```

   - **Apply Remote configuration**
     ```shell
     curl -X POST \
     -H "Authorization: Bearer $(gcloud auth print-access-token)" \
     -H "Content-Type: application/json; charset=utf-8" \
     -d @request.json \
     "https://$REGION-aiplatform.googleapis.com/v1beta1/projects/$PROJECT_ID/locations/$REGION/publishers/google/models/${MODEL_ID}:setPublisherModelConfig"
     ```

   - **Test applied configuration**

       ```shell
       curl -X GET \
       -H "Authorization: Bearer $(gcloud auth print-access-token)" \
       "https://$REGION-aiplatform.googleapis.com/v1beta1/projects/$PROJECT_ID/locations/$REGION/publishers/google/models/${MODEL_ID}:fetchPublisherModelConfig"
       ```

7. Setup Query Logging Table for Retrieval

> **Model-Specific Configurations**: You can use separate existing `.env` files for different models (e.g., `.env-2.5pro`, `.env-2.5flash`).

> [!NOTE]
> **Multi-Table Support**: You can query multiple BigQuery tables simultaneously by providing a comma-separated list:
> ```bash
> export TABLE_ID="2p5-pro, 2p5-flash, 1p5-pro"
> ```
> When multiple tables are specified, the system automatically combines data from all tables using SQL UNION ALL operations.













### Install Libraries

```shell
python -m venv .venv
source .venv/bin/activate
```

```shell
pip install -r requirements.txt
```


Authenticate:
```shell
gcloud auth application-default login
gcloud auth login
```


---

## Quick start

Run load generator:
```shell
python load_generator.py all
```

Generate Summary report of the LLM performance:
```shell
./run_analysis.sh
```

### Monitor Progress
You can monitor the analysis progress in real-time:
```shell
# View script output (including agent thoughts)
tail -f latest_script.log

# View detailed agent logs (tool calls, API requests)
tail -f latest_agent.log
```

Check BigQuery table for the logged llm calls
Check generated .md report inside reports directory (e.g., `reports/latency_report_<timestamp>.md`)

## Load Generator

A tool to generate synthetic load for latency analysis, now powered by ADK agents for better tracking.

### Features
- **ADK Agent Integration**: Uses `google.adk.agents.LlmAgent` to wrap requests, ensuring they are properly logged with agent labels.
- **Direct Client Support**: Option to bypass the agent and use `google.genai.Client` directly for baseline comparison.
- **Configurable Scenarios**: Define scenarios in `load_scenarios.json` with specific prompts, token counts, and concurrency.
- **Latency Tracking**: Logs Time-to-First-Token (TTFT) and End-to-End (E2E) latency.

Uses configurations setup in `load_scenarios.json` file.

### Usage

**Using the Wrapper Script (Recommended):**

For easy switching between different model configurations, use the `load_generator.sh` wrapper script:

```shell
# Run all scenarios with Gemini 2.5 Pro configuration (defaults to 'all' scenario)
./load_generator.sh 2.5pro

# Run all scenarios with Gemini 2.5 Flash configuration
./load_generator.sh 2.5flash

# Run specific scenario with custom model config
./load_generator.sh 2.0flash thinking_vs_baseline

# Run with count override
./load_generator.sh 3pro all --count 5
```

The wrapper script automatically loads the corresponding `.env` file (e.g., `.env-2.5pro`, `.env-2.5flash`) which contains model-specific settings like `MODEL_ID`, generation config defaults, etc.

**Direct Python Script Usage:**

**Run All scenarios:**
```shell
python load_generator.py all
```

**Run with specific .env file:**
```shell
python load_generator.py all --env-file .env-2.5pro
python load_generator.py all --env-file .env-2.5flash
```

**Run with direct client (bypass agent):**
```shell
python3 load_generator.py direct_baseline
```

**Test GenerationConfig variations:**
```shell
# Test low temperature (deterministic)
python load_generator.py config_temp_low

# Test medium temperature (balanced creativity)
python load_generator.py config_temp_medium

# Test high temperature (maximum creativity)
python load_generator.py config_temp_high

# Test conservative maxOutputTokens
python load_generator.py config_tokens_conservative

# Test generous maxOutputTokens (detect wasteful configs)
python load_generator.py config_tokens_generous
```

The new config scenarios help you:
- **Identify optimal temperature settings** for different use cases
- **Detect wasteful maxOutputTokens** configurations 
- **Compare latency impact** of different generation parameters
- **Optimize token efficiency** by finding right-sized limits
- **Test different models** by loading model-specific .env files

---


## Latency Analyzer Agent

The `latency_analyzer` agent is a comprehensive AI-powered tool that automates latency analysis. It combines statistical analysis, pattern detection, and individual query investigation into a single unified agent.

### Features

- **16 Specialized Tools**: Complete coverage of latency analysis needs
- **Automated Insights**: LLM-powered pattern detection and root cause analysis
- **Hypothesis-Driven Research**: Systematic testing of performance hypotheses
- **Cost Tracking**: Token usage analysis and cost estimation
- **Performance Degradation Detection**: Identifies if latency is increasing over time using moving averages.
- **TPOT Analysis**: Calculates Time Per Output Token to distinguish between compute bottlenecks and verbose output.
- **Cost Analysis**: Estimates token costs and identifies expensive query patterns.
- **Individual Query Deep-Dive**: Fetches full details of specific slow queries for root cause analysis.
- **Agent Comparison**: Compares performance across different agents (latency, volume, errors)
- **Model Comparison**: Compares performance across different models (if multiple tables/models are configured)
- **Generation Config Analysis**: Analyzes impact of temperature and maxOutputTokens on latency
- **Per-Agent Breakdown**: Automatically analyzes performance per agent when running global analysis.

### Analysis Tools

**Core Statistics:**
- `get_overall_statistics()` - Mean, median, p90/p95/p99 latency, token stats
- `get_latency_distribution()` - Distribution across latency buckets
- `get_hourly_patterns()` - Time-based patterns (peak hours, working vs weekend)
- `get_agent_comparison()` - Per-agent performance comparison

**Deep Analysis & Research:**
- `analyze_correlation_detailed()` - **Enhanced correlation** including latency vs output+thought tokens with quartile analysis
- `cluster_slow_queries()` - **Pattern detection** by grouping similar slow queries
- `get_token_correlation()` - Latency vs token count correlation
- `get_outlier_analysis()` - Anomaly detection
- `get_slowest_queries()` - Top N slowest queries
- `get_query_details()` - Full details for specific request_id
- `get_concurrent_request_impact()` - Concurrency impact on latency

**Individual Query Analysis (Token-Efficient):**
- `fetch_slow_queries()` - Fetch only request IDs and latency (lightweight, avoids token limits)
- `fetch_single_query()` - Fetch full request/response content for deep analysis
  - **Use case**: Analyze actual content of slow queries (prompts, responses, tool calls)
  - **Pattern**: First call `fetch_slow_queries(10)`, then `fetch_single_query(request_id)` for each
  - **Benefit**: Avoids token limits when dealing with massive payloads

**Advanced Insights:**
- `detect_performance_degradation()` - Trend analysis over time
- `get_cost_analysis()` - Token usage and cost breakdown
- `compare_time_periods()` - Before/after comparison

**GenerationConfig Analysis:**
*   `get_generation_config_comparison()` - Compare latency across different temperature and maxOutputTokens settings
*   `analyze_config_correlation()` - Analyze correlation between config parameters (temperature, maxOutputTokens, topK, topP) and latency
*   `get_config_outliers()` - Identify wasteful configurations (e.g., maxOutputTokens >> actual output) with optimization recommendations

**Model Analysis:**
*   `get_model_comparison()` - Compare KPI metrics across different models (requires multiple tables/models in investigation)
*   `get_agent_model_matrix()` - detailed breakdown of agent performance per model

### Parallel Latency Analyzer (New Swarm Architecture)

The system now includes a **Parallel "Swarm" Architecture** (`agents/parallel_latency_analyzer`) that significantly speeds up analysis by running investigations concurrently.

**Key Features:**
- **Swarm Architecture**: Spawns multiple specialized teams (Strategist, Investigator, Critique, Writer) to analyze different dimensions simultaneously.
- **Context Caching**: Uses ADK Context Caching (`run_with_caching.py`) to reduce TTFT and costs.
- **Dimensions Analyzed**:
    - KPI Compliance
    - Hourly & Daily Patterns
    - Token Correlation & Cost Efficiency
    - Micro-Bursts & Queuing
    - Comparative Analysis (Agents & Models)
    - Slow Query Deep Dive

### Usage

**Run Parallel Analysis (Recommended):**
```shell
./run_test_analysis.sh
```
This script uses `run_with_caching.py` to execute the swarm agent with context caching enabled.

**Run Standard/Old Analysis:**
```shell
./run_autonomous_analysis.sh
```


**Quick Start - Autonomous Analysis (Recommended):**

The **autonomous analysis** automatically includes deep research triggers. When critical issues are detected (KPI failures, strong correlations, dominant clusters), the agent automatically performs deeper investigation:

```shell
./run_autonomous_analysis.sh
```

This workflow adapts its depth based on what it finds, ensuring the most thorough analysis without manual intervention.

**Configuration Format:**

The agent uses JSON config files with two key sections:
1. **`config`**: Parameters for the analysis (time range, KPI targets, scope, etc.)
2. **`queries`**: A concise instruction (1-2 sentences) describing the analysis focus

**Example** (`autonomous_analysis_90d.json`):
```json
{
    "state": {},
    "config": {
        "time_period_days": "90d",
        "analysis_scope": "autonomous",
        "kpis": {
            "mean_latency_target": 3.0,
            "p95_latency_target": 5.0
        },
        "num_slowest_queries": 20,
        "agent_name": null
    },
    "queries": [
        "Perform autonomous latency analysis for the configured time period."
    ]
}
```

**Configuration Parameters:**
- **`time_period_days`**: Analysis window (e.g., "24h", "7d", "90d", "last 27 days", "all")
  > [!WARNING]
  > Using `"time_period_days": "all"` can be very slow and may cause BigQuery operations to time out, as it queries the entire data history. Not recommended for routine analysis.
  
- **`analysis_scope`**: Determines the workflow depth
  - `"standard"`: Quick health check (KPI compliance, basic correlation, patterns)
  - `"autonomous"`: **Comprehensive analysis with automatic deep research triggers** (recommended)
    - Automatically activates deep research when:
      - KPIs fail (mean or P95 above targets)
      - Strong correlations detected (r > 0.7)
      - Dominant clusters found (>30% of slow queries)
      - High variance outliers (std/mean > 0.5)
      - Significant performance degradation (>20%)
      - Request queuing detected (burst correlation > 0.6)
  - `"deep_research"`: Forces deep research mode for all hypotheses (use for manual control)
  
- **`kpis`**: Target latency values
  - `mean_latency_target`: Target for mean latency in seconds (e.g., 3.0)
  - `p95_latency_target`: Target for P95 latency in seconds (e.g., 5.0)
  
- **`agent_name`**: Filter for a specific agent (e.g., "my_agent") or `null` for all agents
- **`filters`**: Granular inclusion/exclusion of agents (New in Parallel Analyzer)
  ```json
  "filters": {
    "agents_included": "latency_analyzer, final_report_assembler", // Only analyze these
    "agents_excluded": "health_check_agent" // Exclude these
  }
  ```

- **`num_slowest_queries`**: Number of slow queries to analyze in detail (e.g., 20)

**How It Works:**

The workflow logic is now in the system prompt (`agents/latency_analyzer/prompts.py`). The agent:
1. Calls `get_analysis_config()` to read the config parameters
2. Selects the appropriate workflow based on `analysis_scope`
3. Follows a systematic analysis approach
4. Generates a comprehensive report
5. Saves the report to `reports/` directory with timestamp

**Report Files:**
- Autonomous analysis (adaptive depth): `reports/latency_report_YYYYMMDD_HHMMSS.md`
- Deep research (forced): `reports/deep_latency_research_report_YYYYMMDD_HHMMSS.md`

Alternatively, run it manually:

```shell
cd agents
adk run --replay ../autonomous_analysis_90d.json latency_analyzer
```

**Output Files:**

The agent generates comprehensive markdown reports in the `reports/` directory:
- **Format:** `reports/latency_report_<timestamp>.md`
- **Content:** Executive summary, hypothesis testing results, key findings, root causes, and prioritized recommendations.


**Interactive Mode:**

```shell
cd agents
adk run latency_analyzer
```

Then ask questions like:
- "Analyze latency for the last 24 hours"
- "Deep research on slow queries with hypothesis testing"
- "Why is performance slow during peak hours?"
- "Compare performance between agent A and agent B"
- "Find the most expensive agents by token usage"
- "Has performance degraded over the last week?"
- "Fetch the 10 slowest queries and analyze each one in detail"
- "Compare latency across different temperature and maxOutputTokens settings"
- "Which generationConfig combination gives the best latency performance?"
- "Are we wasting tokens with over-provisioned maxOutputTokens?"
- "Analyze correlation between temperature and latency"
- "Compare performance between Gemini 1.5 Pro and 2.5 Pro"
- "Which model gives the best latency for the 'search_tool' agent?"


---

## Visualization and Charts (Alternative Approach)

If you prefer visual analysis over AI-driven reports, you can generate comprehensive PDF reports with charts and visualizations using `gemini_logs.py`.

### Generate Analysis Charts

Usage examples:
```shell
# Last 7 days 
# Basic usage with default settings (5min, 10min buckets)
python gemini_logs.py -d 7

# Custom bucket sizes for different data densities
python gemini_logs.py -d 7 -b "600,1800"  # 10min, 30min buckets

# Multiple bucket sizes for comprehensive analysis
python gemini_logs.py -d 7 -b "60,300,600" -m start_time

# Specific time range with custom buckets
python gemini_logs.py -s "2024-01-01 00:00:00" -e "2024-01-02 00:00:00" -b "300,900"
```

### Generated Charts and Visualizations

The script generates comprehensive PDF reports with the following visualizations for each model and agent:

#### 1. Agent Summary Analysis
- **Summary Statistics Table**: Per-agent breakdown (calls, mean latency, P95/P99)
- **Total LLM Calls per Agent**: Bar chart showing request volume
- **Mean Latency per Agent**: Bar chart with error bars

#### 2. Latency Distribution Analysis
- **Latency Distribution Histogram**: Frequency distribution
- **Cumulative Distribution**: Percentage of requests by latency threshold
- **Box Plot**: Quartiles, median, and outliers
- **Statistical Summary**: Mean, median, standard deviation, percentiles

#### 3. Latency vs Output Tokens
Four scatter plots with different scale combinations:
- Linear-Linear, Log-Linear, Linear-Log, Log-Log
- Color mapping based on input tokens
- Correlation coefficient and trend line

#### 4. Latency vs Input Tokens
Similar to output tokens analysis with color mapping based on output tokens

#### 5. Latency vs Output+Thought Tokens
Analyzes combined impact of output and thought tokens on latency

#### 6. Hourly Analysis by Day Type
- Request count by hour (working vs non-working days)
- Mean latency by hour
- Box plots by day of week
- Working vs non-working day comparison

### Output Files
- **PDF Report**: `reports/complete_analysis_<model_name>__<timestamp>.pdf`
- **PNG Files**: High-resolution charts in `reports/png/` directory (300 DPI and 4K)

---

## Troubleshooting

### Debugging Agent Tool Errors

If the agent encounters errors during analysis, follow these steps:

#### 1. Check Agent Logs

The agent writes detailed logs to a temporary directory. The log path is shown when you start the agent:

```bash
# View the latest log in real-time
tail -F /var/folders/.../agents_log/agent.latest.log

# View last 100 lines
tail -100 /var/folders/.../agents_log/agent.latest.log

# Search for errors
grep -i error /var/folders/.../agents_log/agent.latest.log
```

**Common log patterns:**
- `ERROR - utils.py:XXXX - Error in <tool_name>: <error_message>` - Tool execution error
- `WARNING - <message>` - Non-fatal issues (e.g., insufficient data for quartile analysis)

#### 2. Test Tools Directly

You can test individual tools in Python to debug issues:

```python
from agents.latency_analyzer.utils import analyze_correlation_detailed, get_overall_statistics

# Test a specific tool
result = analyze_correlation_detailed(time_range="7d")
print(result)

# Check for errors in the JSON response
import json
data = json.loads(result)
if "error" in data:
    print(f"Error: {data['error']}")
```

#### 3. Common Issues and Solutions

**Issue: "Insufficient data for correlation analysis"**
- **Cause:** Not enough data points in the specified time range
- **Solution:** Increase the time range (e.g., from "24h" to "7d" or "30d")

**Issue: "Bin labels must be one fewer than the number of bin edges"**
- **Cause:** Too few unique token values to create quartiles
- **Solution:** This is now handled gracefully with a warning. The analysis will continue without quartile breakdown.

**Issue: "No data found"**
- **Cause:** No logs in BigQuery for the specified time range or filters
- **Solution:** 
  - Verify logs are being written to BigQuery
  - Check environment variables (PROJECT_ID, DATASET_ID, TABLE_ID)
  - Verify the time range has data: `bq query "SELECT COUNT(*) FROM \`${PROJECT_ID}.${DATASET_ID}.${TABLE_ID}\`"`

**Issue: "PROJECT_ID environment variable is not set"**
- **Cause:** Environment variables not loaded
- **Solution:** 
  - Ensure `.env` file exists and is properly formatted
  - Run `source .env` before starting the agent
  - Check that the agent is loading the `.env` file (you should see "Loaded .env file" in logs)

#### 4. Enable Verbose Logging

To get more detailed logs, you can modify the logging level in the agent:

```python
# In agents/latency_analyzer/agent.py or utils.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

#### 5. Verify BigQuery Access

Test that you can query the logs table:

```bash
# Test query
bq query --use_legacy_sql=false "
SELECT COUNT(*) as total_requests,
       MIN(logging_time) as earliest,
       MAX(logging_time) as latest
FROM \`${PROJECT_ID}.${DATASET_ID}.${TABLE_ID}\`
"
```

Expected output should show:
- `total_requests` > 0
- `earliest` and `latest` timestamps

#### 6. Check Tool Return Format

All tools return JSON strings. If a tool fails, it returns:
```json
{"error": "Error message describing what went wrong"}
```

The agent should detect these errors and report them to you.

---

## Verification

### Automated Tests

The project now includes a comprehensive test suite in `tests/`:

```shell
# Run all tests
python -m pytest tests/

# Key Test Modules:
# - tests/test_model_analysis.py: Verifies model comparison logic
# - tests/test_slowest_queries.py: Checks slow query extraction and sanitization
# - tests/test_tool_fixes.py: Verifies fixes for common tool errors
# - tests/test_agent_filtering_sql.py: Tests the SQL generation for agent filtering
```

To verify the agent's logic (using mocks), run the test script:

```shell
python3 tests/test_slow_query_analyzer.py
```

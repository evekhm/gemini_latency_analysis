# Latency Analysis

## Overview

An AI-powered performance analytics platform that automatically analyzes LLM application latency, identifies bottlenecks, and provides actionable optimization recommendations.

**What it does:**
- Analyzes BigQuery logs of LLM calls to identify why queries are slow
- Uses AI-driven hypothesis testing to find root causes (token size, agent design, concurrency, time patterns)
- Clusters similar slow queries to identify systemic issues
- Tracks token usage and estimates costs by agent
- Provides prioritized recommendations with expected impact

**Key benefits:**
- Replaces manual chart analysis with automated reports
- Identifies optimization opportunities to reduce token usage
- Pinpoints specific actions to improve latency
- Detects performance degradation trends before they impact users

**Quick start:**
```bash
./run_latency_research.sh  # One command for comprehensive 90-day analysis
```

---

## Prerequisites

### Enable Gemini Requests Logging
Vertex AI can log samples of requests and responses for Gemini and supported partner models.
The logs are saved to a BigQuery table for viewing and analysis.

To enable logging, follow these [instructions](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/request-response-logging)


Needs to be done to each model being used:
```shell
MODEL="gemini-2.0-flash-lite"
#MODEL="gemini-2.5-pro"
# MODEL="gemini-2.0-flash"
curl -X POST \
-H "Authorization: Bearer $(gcloud auth print-access-token)" \
-H "Content-Type: application/json; charset=utf-8" \
-d @request.json \
"https://$REGION-aiplatform.googleapis.com/v1beta1/projects/$PROJECT_ID/locations/$REGION/publishers/google/models/$MODEL:setPublisherModelConfig"
```

> request.json
```json
{
  "publisherModelConfig": {
     "loggingConfig": {
       "enabled": true,
       "samplingRate": 1,
       "bigqueryDestination": {
         "outputUri": "bq://PROJECT_ID.DATASET.TABLENAME"
       },
       "enableOtelLogging": true
     }
   }
 }
```

### Install libraries

```shell
python -m venv .venv
source .venv/bin/activate
```

Install libraries:

```shell
pip install -r requirements.txt
```

### Set environment variables

Update environment variables in .env file accordingly:
```shell
export PROJECT_ID="..."
export DATASET="..."           # configured for logging
export GEMINI_LOG_TABLE="..."  # configured for logging
```

Authenticate:
```shell
gcloud auth application-default login
gcloud auth login
```

## Generate Analysis of Gemini Level Logs

Usage examples:
```shell
# Last 7 days 
# Basic usage with default settings (5min, 10min buckets)
python gemini_analysis.py -d 7

# Custom bucket sizes for different data densities
python gemini_logs.py -d 7 -b "600,1800"  # 10min, 30min buckets

# Multiple bucket sizes for comprehensive analysis
python gemini_logs.py -d 7 -b "60,300,600" -m start_time

# Specific time range with custom buckets
python gemini_logs.py -s "2024-01-01 00:00:00" -e "2024-01-02 00:00:00" -b "300,900"
```


```shell
python gemini_logs.py --start "2025-08-14 00:00:00" --end "2025-08-20 23:59:59"
```

## Generated Charts and Visualizations

The `gemini_logs.py` script generates comprehensive PDF reports with the following visualizations for each model and agent:

### 1. Agent Summary Analysis
- **Summary Statistics Table**: Shows per-agent breakdown including:
  - Total calls
  - Mean latency with standard deviation
  - P95 and P99 latency percentiles
- **Total LLM Calls per Agent**: Bar chart showing request volume by agent
- **Mean Latency per Agent**: Bar chart with error bars showing average latency and variability

### 2. Latency Distribution Analysis
- **Latency Distribution Histogram**: Shows the frequency distribution of request latencies
- **Cumulative Distribution**: Displays what percentage of requests complete within various latency thresholds
- **Box Plot**: Visualizes latency quartiles, median, and outliers
- **Statistical Summary**: Key metrics including mean, median, standard deviation, and percentiles

### 3. Latency vs Output Tokens
Four scatter plots showing the relationship between latency and output token count with different scale combinations:
- **Linear-Linear**: Standard view for overall patterns
- **Log-Linear**: Logarithmic x-axis (tokens), linear y-axis (latency)
- **Linear-Log**: Linear x-axis (tokens), logarithmic y-axis (latency)
- **Log-Log**: Both axes logarithmic for power-law relationships

Each plot includes:
- Color mapping based on input tokens (when available)
- Correlation coefficient
- Trend line
- Statistical summary (N, token ranges, latency range)

### 4. Latency vs Input Tokens
Similar to output tokens, but analyzing the relationship between latency and input token count:
- Four scale combinations (Linear-Linear, Log-Linear, Linear-Log, Log-Log)
- Color mapping based on output tokens (when available)
- Correlation analysis and trend lines

### 5. Latency vs Output+Thought Tokens (NEW)
Analyzes the combined impact of output tokens and thought tokens on latency:
- Four scale combinations for comprehensive analysis
- Color mapping based on input tokens
- Useful for understanding the total generation cost including reasoning tokens
- Correlation and trend analysis

### 6. Hourly Analysis by Day Type
Breaks down latency patterns by time of day and working vs. non-working days:
- **Request Count by Hour**: Separate charts for working days and non-working days
- **Mean Latency by Hour**: Shows how latency varies throughout the day
- **Box Plots by Day of Week**: Latency distribution for each day of the week
- **Comparison Chart**: Direct comparison of working vs. non-working day patterns

### Output Files
- **PDF Report**: `out/complete_analysis_<model_name>__<timestamp>.pdf` - Contains all visualizations and terminal output
- **PNG Files**: High-resolution charts saved to `out/png/` directory:
  - Standard resolution (300 DPI)
  - 4K resolution (400 DPI) for presentations
  - Individual files for each chart type

## Latency Analyzer Agent

The `latency_analyzer` agent is a comprehensive LLM performance analysis tool that automates the analysis currently done manually by inspecting charts from `gemini_logs.py`. It provides deep insights into latency patterns, identifies bottlenecks, and delivers actionable optimization recommendations.

### Features

- **Comprehensive Analysis**: 12 specialized tools covering all aspects of latency analysis
- **Automated Insights**: LLM-powered pattern detection and root cause analysis
- **Cost Tracking**: Token usage analysis and cost estimation
- **Trend Detection**: Identifies performance degradation over time
- **Agent Comparison**: Compares performance across different agents
- **Deep Dive**: Analyzes individual slow queries in detail

### Analysis Tools

**Core Statistics:**
- `get_overall_statistics()` - Mean, median, p90/p95/p99 latency, token stats
- `get_latency_distribution()` - Distribution across latency buckets
- `get_hourly_patterns()` - Time-based patterns (peak hours, working vs weekend)
- `get_agent_comparison()` - Per-agent performance comparison

**Deep Analysis:**
- `analyze_correlation_detailed()` - **Enhanced correlation** including latency vs output+thought tokens with quartile analysis
- `cluster_slow_queries()` - **Pattern detection** by grouping similar slow queries
- `get_token_correlation()` - Latency vs token count correlation
- `get_outlier_analysis()` - Anomaly detection
- `get_slowest_queries()` - Top N slowest queries
- `get_query_details()` - Full details for specific request_id
- `get_concurrent_request_impact()` - Concurrency impact on latency

**Advanced Insights:**
- `detect_performance_degradation()` - Trend analysis over time
- `get_cost_analysis()` - Token usage and cost breakdown
- `compare_time_periods()` - Before/after comparison

### Deep Research Mode

The agent supports **hypothesis-driven research analysis** for thorough investigation:

**Trigger with phrases like:**
- "Deep analysis of latency patterns"
- "Research all possible causes"
- "Test hypotheses about performance"
- "Find all interesting patterns"

**Research Process:**
1. **Hypothesis Generation** - Agent generates multiple hypotheses to test
2. **Systematic Testing** - Tests each hypothesis with appropriate tools
3. **Findings Synthesis** - Summarizes accepted/rejected/inconclusive hypotheses
4. **Follow-Up Questions** - Suggests specific next steps based on findings

**Example Research Hypotheses:**
- H1: Output+thought tokens drive latency
- H2: Specific agents have systematic issues
- H3: Time-based patterns exist
- H4: Slow queries cluster into distinct groups
- H5: Outliers share common characteristics

**Key Research Tools:**
- `analyze_correlation_detailed()` - Tests token correlation hypotheses with statistical rigor
- `cluster_slow_queries()` - Identifies patterns and groups similar queries

The agent will provide:
- Statistical evidence for each hypothesis
- Clustering breakdown with similarities
- Correlation analysis (including output+thought tokens)
- Specific follow-up questions for deeper investigation


### Usage

**Interactive Mode:**

```shell
cd agents
adk run latency_analyzer
```

Then ask questions like:
- "Analyze latency for the last 24 hours"
- "Why is performance slow during peak hours?"
- "Compare performance between agent A and agent B"
- "Find the most expensive agents by token usage"
- "Has performance degraded over the last week?"

**Automated Deep Research (Recommended):**

For comprehensive 90-day analysis, use the provided script:

```shell
./run_latency_research.sh
```

This runs a complete hypothesis-driven research analysis including:
- Token correlation analysis (output+thought tokens)
- Clustering of slow queries
- Agent performance comparison
- Performance degradation trends
- Cost analysis
- Prioritized recommendations
- Follow-up questions

**Non-Interactive Mode (Custom Replay):**

Create a JSON file with your queries:

```json
{
  "state": {},
  "queries": [
    "Analyze latency for the last 7 days",
    "Deep analysis of latency patterns"
  ]
}
```

Then run:
```shell
cd agents
adk run --replay ../your_queries.json latency_analyzer
```

**Example Replay Files:**
- `deep_latency_research_90d.json` - Comprehensive 90-day research analysis
- `auto_analysis_input.json` - Simple slow query analysis (for slow_query_analyzer)

**Example Analysis:**

The agent will:
1. Gather statistics using appropriate tools
2. Identify patterns and anomalies
3. Perform root cause analysis
4. Generate a structured report with:
   - Executive summary
   - Key findings with data evidence
   - Root cause explanations
   - Prioritized recommendations

### Environment Setup

Uses the same `.env` file as other agents (see main Environment Setup section).

### How It Works

Instead of generating charts, the agent:
1. Queries BigQuery for structured data (statistics, correlations, distributions)
2. Analyzes the data using LLM reasoning
3. Identifies patterns and bottlenecks
4. Provides specific, actionable recommendations

This approach is more precise than visual chart interpretation and enables automated analysis.



## Slow Query Analyzer Agent

The `slow_query_analyzer` agent analyzes slow queries from BigQuery logs to identify performance bottlenecks and optimization opportunities.

### Features

- **Token-Efficient Processing**: Fetches query metadata first, then retrieves full details individually to avoid exceeding LLM token limits
- **Comprehensive Analysis**: For each slow query, analyzes:
  - Context (what the query is about based on request/response content)
  - Latency drivers (massive input, verbose output, long reasoning, tool latency)
  - Root causes with specific token counts
- **Pattern Detection**: Identifies clusters of similar slow queries
- **Actionable Recommendations**: Provides specific optimization suggestions (prompt distillation, contextual scoping, etc.)

### Environment Setup

The agent requires environment variables to be configured in the main project `.env` file:

**`.env`** (at project root: `/Users/evekhm/projects/adk/latency_analysis/.env`):
```
PROJECT_ID=your-gcp-project
DATASET=your-bigquery-dataset
GEMINI_LOG_TABLE=your-log-table
MODEL=gemini-2.5-pro
GOOGLE_GENAI_USE_VERTEXAI=TRUE
GOOGLE_CLOUD_LOCATION=us-central1
```

The ADK CLI will automatically load these environment variables when running the agent.

### How It Works

The agent uses a two-phase approach to avoid token limits:
1. **Phase 1**: Calls `fetch_slow_queries` to get lightweight metadata (request_id + latency) for top 10 queries
2. **Phase 2**: For each request_id, calls `fetch_single_query` to get full details and analyzes individually

This allows processing of queries with massive `full_request` and `full_response` fields without exceeding the 1M token limit.

### Usage

**Interactive Mode**

1. Start the agent:
   ```shell
   cd agents
   adk run slow_query_analyzer
   ```

2. When prompted with `[user]:`, type:
   ```
   Analyze the slow queries
   ```

3. The agent will:
   - Fetch the top 10 slowest queries
   - Analyze each query individually
   - Generate a comprehensive report with patterns and recommendations

**Non-Interactive Mode (using replay file)**

```shell
cd agents
adk run --replay ../auto_analysis_input.json slow_query_analyzer
```

This automatically executes the analysis without requiring user input.

### Verification

To verify the agent's logic (using mocks), run the test script:

```shell
python3 tests/test_slow_query_analyzer.py
```

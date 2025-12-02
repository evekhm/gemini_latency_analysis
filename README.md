<!-- TOC -->
* [Latency Analysis](#latency-analysis)
  * [Overview](#overview)
  * [Prerequisites](#prerequisites)
    * [Enable Gemini Requests Logging](#enable-gemini-requests-logging)
    * [Install Libraries](#install-libraries)
    * [Set Environment Variables](#set-environment-variables)
  * [Latency Analyzer Agent (Recommended Approach)](#latency-analyzer-agent-recommended-approach)
    * [Features](#features)
    * [Analysis Tools](#analysis-tools)
    * [Usage](#usage)
    * [Deep Research Mode](#deep-research-mode)
    * [How It Works](#how-it-works)
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
  * [Verification](#verification)
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

### Set Environment Variables

Update environment variables in [.env](.env) file accordingly:
```shell
export PROJECT_ID="..."
export MODEL="gemini-2.5-pro"  # Gemini Model for which configuration is applied. You will need to re-apply this step for each Gemini model being used, e.g. for flash, pro, etc. separately.
export DATASET="..."           # name of the dataset, configured for logging. Make sure to create such dataset first.
export GEMINI_LOG_TABLE="..."  # name of the table configured for logging. You want each MODEL to have its own table. The table will be created automatically.
```

Load environment variables:
```shell
source .env
```

### Enable Gemini Requests Logging

Vertex AI can log samples of requests and responses for Gemini and supported partner models.
The logs are saved to a BigQuery table for viewing and analysis.

To enable logging, follow these [instructions](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/request-response-logging)


**(Optional) Create dataset**
```shell
bq --location="$GOOGLE_CLOUD_LOCATION" mk --dataset --description "Dataset for LLM logging" --project_id=$PROJECT_ID ${DATASET} || echo "Dataset DATASET already exists."
```

**Create `request.json` file**
```shell
cat > request.json <<EOF
{
  "publisherModelConfig": {
     "loggingConfig": {
       "enabled": true,
       "samplingRate": 1,
       "bigqueryDestination": {
         "outputUri": "bq://${PROJECT_ID}.${DATASET}.${GEMINI_LOG_TABLE}"
       },
       "enableOtelLogging": true
     }
   }
 }
EOF
```

**Apply Remote configuration** 
```shell
curl -X POST \
-H "Authorization: Bearer $(gcloud auth print-access-token)" \
-H "Content-Type: application/json; charset=utf-8" \
-d @request.json \
"https://$REGION-aiplatform.googleapis.com/v1beta1/projects/$PROJECT_ID/locations/$REGION/publishers/google/models/$MODEL:setPublisherModelConfig"
```


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

## Latency Analyzer Agent (Recommended Approach)

The `latency_analyzer` agent is a comprehensive AI-powered tool that automates latency analysis. It combines statistical analysis, pattern detection, and individual query investigation into a single unified agent.

### Features

- **16 Specialized Tools**: Complete coverage of latency analysis needs
- **Automated Insights**: LLM-powered pattern detection and root cause analysis
- **Hypothesis-Driven Research**: Systematic testing of performance hypotheses
- **Cost Tracking**: Token usage analysis and cost estimation
- **Trend Detection**: Identifies performance degradation over time
- **Agent Comparison**: Compares performance across different agents
- **Individual Query Analysis**: Deep-dive into specific slow queries with full request/response content

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

### Usage

**Quick Start - Automated Deep Research (Recommended):**

For comprehensive 90-day analysis with hypothesis testing:

```shell
./run_comprehensive_analysis.sh
```

This script uses the `comprehensive_analysis_90d.json` replay file and runs a complete analysis including:
- Overall statistics and health check
- Token correlation analysis (output+thought tokens)
- Clustering of slow queries
- Agent performance comparison
- Performance degradation trends
- Cost analysis
- Individual slow query deep-dives
- Prioritized recommendations
- Follow-up questions

Alternatively, you can run it manually:

```shell
cd agents
adk run --replay ../comprehensive_analysis_90d.json latency_analyzer
```

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

**Custom Replay Mode:**

Create a JSON file with your queries:

```json
{
  "state": {},
  "queries": [
    "Analyze latency for the last 7 days",
    "Deep analysis of latency patterns",
    "Fetch the slowest queries and analyze them individually"
  ]
}
```

Then run:
```shell
cd agents
adk run --replay ../your_queries.json latency_analyzer
```

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
- H6: Individual slow queries reveal specific bottlenecks

**Key Research Tools:**
- `analyze_correlation_detailed()` - Tests token correlation hypotheses with statistical rigor
- `cluster_slow_queries()` - Identifies patterns and groups similar queries
- `fetch_slow_queries()` + `fetch_single_query()` - Deep-dive into individual query content

The agent will provide:
- Statistical evidence for each hypothesis
- Clustering breakdown with similarities
- Correlation analysis (including output+thought tokens)
- Individual query analysis with root causes
- Specific follow-up questions for deeper investigation

### How It Works

Instead of generating charts, the agent:
1. Queries BigQuery for structured data (statistics, correlations, distributions)
2. Analyzes the data using LLM reasoning
3. Identifies patterns and bottlenecks
4. Fetches individual slow queries when needed for deep analysis
5. Provides specific, actionable recommendations

This approach is more precise than visual chart interpretation and enables automated analysis.

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
- **PDF Report**: `out/complete_analysis_<model_name>__<timestamp>.pdf`
- **PNG Files**: High-resolution charts in `out/png/` directory (300 DPI and 4K)

---

## Verification

To verify the agent's logic (using mocks), run the test script:

```shell
python3 tests/test_slow_query_analyzer.py
```

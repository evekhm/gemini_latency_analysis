# Latency Analysis

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





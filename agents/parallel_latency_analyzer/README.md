# Parallel Latency Analyzer (Swarm Architecture)

This agent implements a **Parallel "Swarm" Architecture** for autonomous latency analysis. 
Instead of processing one dimension at a time, it spawns multiple specialized teams to investigate different aspects of system performance simultaneously, significantly reducing analysis time.

## Architecture

- **Root Agent**: `parallel_latency_analyzer`
- **Orchestrator**: `complete_report_generator`
- **Swarm**: `investigation_swarm` (ParallelAgent)
- **Deep Research**: Triggered autonomously when hypotheses are blocked or need validation.

### The Teams
Each dimension is handled by a dedicated `SequentialAgent` pipeline ("Team") consisting of:
1.  **Strategist**: PLANS the investigation (Generates specific questions & hypotheses).
2.  **Investigator**: DOES the work (Runs BigQuery tools).
3.  **Critique**: REVIEWS the findings (Hostile reviewer loop).
4.  **Escalator**: CHECKS if critique passed or failed.
5.  **Writer**: SUMMARIZES the findings into a report section.

### Dimensions Analyzed
1.  **KPI Compliance**: Mean/P95 vs Targets (PASS/FAIL).
2.  **Hourly Patterns**: Peak hours, working vs weekend.
3.  **Token Correlation**: Input/Output and Thought impact on latency (H1). Includes Thought Tokens analysis.
4.  **Micro-Bursts**: Queuing detection (H6).
5.  **Agent Comparison**: Performance by agent (H2).
6.  **Slow Queries**: Deep dive into top outliers and clusters (H4, H5).
7.  **Cost & Efficiency**: Token velocity (TPOT) and compute inefficiency.
8.  **Model Performance**: Impact of model choice (H4).

## Configuration

The agent is configured via a JSON file (e.g., `autonomous_analysis_90d.json`):

```json
{
  "time_period": "last 90 days",
  "kpis": {
    "mean_latency_target": 3.0,
    "p95_latency_target": 5.0
  },
  "agent_name": null, // Set to string to filter analysis for one agent
  "num_slowest_queries": 20,
  "analysis_scope": "autonomous"
}
```

## Environment Variables

Ensure `.env` contains:
- `PROJECT_ID`: Google Cloud Project ID
- `DATASET_ID`: BigQuery Dataset
- `AGENT_TABLE_ID`: Comma-separated list of BigQuery tables to analyze
- `AGENT_MODEL_ID`: Model to use for the agent (e.g., `gemini-1.5-pro-002`)

## Usage

### Running the Analysis
The recommended way to run the analysis is using the provided shell script:

```bash
./run_autonomous_analysis2.sh
```

This script handles:
1.  Environment setup and gRPC noise suppression.
2.  Loading variables from `.env`.
3.  Invoking the ADK runner with the correct configuration.

### Tools Available
The agent has access to a comprehensive suite of BigQuery tools in `utils.py`, including:
- `get_overall_statistics`
- `get_hourly_patterns`
- `analyze_correlation_detailed`
- `cluster_slow_queries`
- `get_slowest_queries`
- `get_query_details` (aliased as `get_request_details`)
- `build_multi_table_source` (handles multi-table querying)

## Troubleshooting

- **"Unrecognized name: T"**: Fixed by ensuring `build_multi_table_source` aliases tables correctly.
- **Missing Tool**: If the agent reports a missing tool, check `utils.py` aliases.
- **gRPC Warnings**: These are suppressed by the runner script but are benign networking logs.

# agent.py - Latency Analyzer Agent
import os
from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.genai import types

from .prompts import PROMPT_LATENCY_ANALYZER
from .utils import (
    get_overall_statistics,
    get_latency_distribution,
    get_hourly_patterns,
    get_agent_comparison,
    get_token_correlation,
    get_outlier_analysis,
    get_slowest_queries,
    get_query_details,
    get_concurrent_request_impact,
    detect_performance_degradation,
    get_cost_analysis,
    compare_time_periods,
    cluster_slow_queries,
    analyze_correlation_detailed,
    # Import slow query tools from slow_query_analyzer
    fetch_slow_queries,
    fetch_single_query,
    fetch_slow_queries_batch,  # Batch fetch for efficiency
    fetch_fastest_queries,     # Baseline comparison
    # Report generation
    save_analysis_report,
    get_analysis_metadata,  # Get actual env values for report headers
    verify_data_access,  # Verify configuration and access
    # New TPOT tool
    get_token_velocity,
    # New KPI and Queuing tools
    analyze_request_queuing,
    check_kpi_compliance,
    get_analysis_config  # Tool to read config
)

__dir__ = os.path.dirname(__file__)
load_dotenv(dotenv_path=os.path.join(__dir__, "../../.env"))

# Get the model from environment variable
MODEL = os.getenv('MODEL')

# Latency Analyzer Agent with all analysis tools (merged with slow_query_analyzer)
latency_analyzer = LlmAgent(
    name="latency_analyzer",
    model=MODEL,
    description="Comprehensive LLM latency and performance analyzer. Analyzes BigQuery logs, identifies bottlenecks, detects patterns, provides actionable optimization recommendations, and performs deep-dive analysis of individual slow queries.",
    instruction=PROMPT_LATENCY_ANALYZER,
    tools=[
        # Core statistics
        get_overall_statistics,
        get_latency_distribution,
        get_hourly_patterns,
        get_agent_comparison,
        # Correlation & patterns
        get_token_correlation,
        analyze_correlation_detailed,  # Enhanced correlation with output+thought
        get_outlier_analysis,
        get_slowest_queries,
        get_query_details,
        cluster_slow_queries,  # Clustering for pattern detection
        get_concurrent_request_impact,
        # Advanced analysis
        detect_performance_degradation,
        get_cost_analysis,
        compare_time_periods,
        # Individual query analysis (from slow_query_analyzer)
        fetch_slow_queries,  # Fetch metadata for slowest queries
        fetch_single_query,  # Fetch full details for a specific query
        fetch_slow_queries_batch,  # Batch fetch multiple queries efficiently
        # Report generation
        save_analysis_report,  # Save final report to markdown file
        get_analysis_metadata,  # Get actual environment metadata
        verify_data_access,    # Verify BigQuery configuration and access
        # TPOT Analysis
        get_token_velocity,   # Analyze generation speed vs volume
        # KPI & Queuing
        analyze_request_queuing, # Detect micro-bursts and queuing
        check_kpi_compliance,     # Check against performance targets
        get_analysis_config      # Read analysis configuration
    ],
    generate_content_config=types.GenerateContentConfig(
        temperature=0,
        max_output_tokens=8192,
        http_options=types.HttpOptions(
            timeout=120000,  # 120 seconds (in ms)
            retry_options=types.HttpRetryOptions(
                attempts=3,  # Retry up to 3 times
                initial_delay=2.0,  # Start with 2 second delay
                max_delay=60.0,  # Max 60 second delay between retries
                exp_base=2.0  # Exponential backoff base
            )
        )
    )
)

# Alias for ADK CLI compatibility
root_agent = latency_analyzer

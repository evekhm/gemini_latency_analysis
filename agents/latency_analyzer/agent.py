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
    # Report generation
    save_analysis_report,
    # New TPOT tool
    get_token_velocity,
    # New KPI and Queuing tools
    analyze_request_queuing,
    check_kpi_compliance
)

__dir__ = os.path.dirname(__file__)
load_dotenv(dotenv_path=os.path.join(__dir__, "../../.env"))

# Get the model from environment variable
MODEL = os.getenv('MODEL', 'gemini-2.0-flash-thinking-exp')

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
        # Report generation
        save_analysis_report,  # Save final report to markdown file
        # TPOT Analysis
        get_token_velocity,   # Analyze generation speed vs volume
        # KPI & Queuing
        analyze_request_queuing, # Detect micro-bursts and queuing
        check_kpi_compliance     # Check against performance targets
    ],
    generate_content_config=types.GenerateContentConfig(
        temperature=0,
        max_output_tokens=8192
    )
)

# Alias for ADK CLI compatibility
root_agent = latency_analyzer

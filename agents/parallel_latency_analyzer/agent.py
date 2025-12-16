import logging
import os
from collections.abc import AsyncGenerator
from typing import Literal, List, Optional

import google.cloud.logging
from dotenv import load_dotenv
from google.adk.agents import BaseAgent
from google.adk.agents import LlmAgent
from google.adk.agents import LoopAgent
from google.adk.agents import ParallelAgent
from google.adk.agents import SequentialAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.genai import types
from pydantic import BaseModel, Field

from .utils import CURRENT_DIMENSION_LIST
from .utils import (
    get_analysis_metadata,
    save_analysis_report,
    trigger_latency_parallel_report,
    process_latency_question,
    read_report_content,
    has_report_content,
    accumulate_investigator_output,
    # Import necessary tools for the Investigator
    get_overall_statistics,
    get_latency_distribution,
    get_hourly_patterns,
    get_agent_comparison,
    get_token_correlation,
    get_outlier_analysis,
    get_slowest_queries,
    get_concurrent_request_impact,
    detect_performance_degradation,
    get_cost_analysis,
    cluster_slow_queries,
    analyze_correlation_detailed,
    get_token_velocity,
    analyze_request_queuing,
    get_query_details,
    get_request_details,
    get_daily_patterns,
    check_kpi_compliance,
    analyze_thinking_overhead,
    detect_compute_inefficiency,
    get_model_comparison,
    get_agent_model_matrix,
    get_generation_config_comparison,
    analyze_config_correlation,
    get_config_outliers,
    get_analysis_config,
    get_hourly_model_distribution,
    get_hourly_model_latency_heatmap,
    # Individual query analysis tools
    fetch_slow_queries,
    fetch_single_query,
    fetch_slow_queries_batch,
    fetch_fastest_queries,
    # Time period comparison
    fetch_fastest_queries,
    # Time period comparison
    compare_time_periods,
    verify_data_access,
    get_tool_usage_report,
    get_subagent_tool_usage
)


from .prompts import (
    ROOT_AGENT_PROMPT,
    STRATEGIST_PROMPT,
    INVESTIGATOR_PROMPT,
    CRITIQUE_PROMPT,
    SECTION_WRITER_PROMPT,
    FINAL_REPORT_ASSEMBLER_PROMPT,
    REPORT_SAVER_PROMPT,
    MARKDOWN_CORRECTOR_PROMPT
)

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../../.env"))

cloud_logging_client = google.cloud.logging.Client()
cloud_logging_client.setup_logging()

from .tracing_model import TracingGemini


MODEL_ID = os.getenv('AGENT_MODEL_ID') or os.getenv('MODEL')
assert MODEL_ID, "MODEL_ID is not set"


MODEL = TracingGemini(
    model=MODEL_ID,
    retry_options=types.HttpRetryOptions(
        initial_delay=1.0,
        attempts=10,
        exp_base=2.0,
        jitter=1.0,
        http_status_codes=[429, 500, 503, 504]
    )
)

CONTENT_CONFIG = types.GenerateContentConfig(
    temperature=0.0, # More deterministic output
    max_output_tokens=65536,
)

# =========================================
# CLASSES & SCHEMAS
# =========================================

class InvestigationEscalationChecker(BaseAgent):
    """Checks investigation evaluation and escalates to stop the loop if grade is 'pass'."""
    feedback_key: str = Field(default="INVESTIGATION_FEEDBACK", description="Session key to check for grade.")

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        evaluation_result = ctx.session.state.get(self.feedback_key)
        if evaluation_result and evaluation_result.get("grade") == "pass":
            logging.info(f"[{self.name}] Investigation evaluation passed. Escalating.")
            yield Event(author=self.name, actions=EventActions(escalate=True))
        else:
            logging.info(f"[{self.name}] Loop continuing.")
            yield Event(author=self.name)

class InvestigationFeedback(BaseModel):
    grade: Literal["pass", "fail"] = Field(description="'pass' if findings are sufficient and evidence-based, 'fail' otherwise.")
    comment: str = Field(description="Detailed critique of gaps or validation.")
    follow_up_questions: Optional[List[str]] = Field(description="Specific directions to fix gaps if fail.")

# =========================================
# PARALLEL FACTORY
# =========================================

def build_dimension_team(dimension_name: str) -> SequentialAgent:
    """Creates a dedicated analysis team for a specific latency dimension."""
    # safe name for unique keys
    safe_name = (
        dimension_name.lower()
        .replace("&", "and")
        .replace("(", "")
        .replace(")", "")
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
    )
    
    KEY_STRAT_OUTPUT = f"{safe_name}_STRAT_OUTPUT"
    KEY_DOC_OUTPUT = f"{safe_name}_DOC_OUTPUT"
    KEY_FEEDBACK = f"{safe_name}_FEEDBACK"
    KEY_FINAL_SECTION = f"SECTION_{safe_name}"

    # 1. Primer Agent
    primer = LlmAgent(
        name=f"primer_{safe_name}",
        model=MODEL,
        description=f"Outputs the focus area: {dimension_name}",
        instruction=f"The designated focus area for this task is '{dimension_name}'. Output ONLY the dimension name.",
        generate_content_config=types.GenerateContentConfig(temperature=0),
        output_key="FOCUS_AREA"
    )

    # 2. Strategist
    strategist = LlmAgent(
        name=f"strategist_{safe_name}",
        model=MODEL,
        description="Generates specific analysis questions",
        instruction=STRATEGIST_PROMPT,
        generate_content_config=CONTENT_CONFIG,
        output_key=KEY_STRAT_OUTPUT
    )

    # 3. Investigator
    investigator_instruction = INVESTIGATOR_PROMPT + f"\nYour input is QUESTIONS = [{{ {KEY_STRAT_OUTPUT} ? }}]\nCRITIQUE_FEEDBACK = [{{ {KEY_FEEDBACK} ? }}]."
    
    investigator = LlmAgent(
        name=f"investigator_{safe_name}",
        model=MODEL,
        description="Executes tools to gather data",
        instruction=investigator_instruction,
        tools=[
            # Core statistics
            get_overall_statistics,
            get_latency_distribution,
            get_hourly_patterns,
            get_daily_patterns,
            get_agent_comparison,
            get_model_comparison,
            get_agent_model_matrix,
            # Correlation & patterns
            get_token_correlation,
            analyze_correlation_detailed,
            get_outlier_analysis,
            get_slowest_queries,
            cluster_slow_queries,
            get_concurrent_request_impact,
            # Advanced analysis
            detect_performance_degradation,
            get_cost_analysis,
            compare_time_periods,
            # Individual query analysis
            get_query_details,
            get_request_details,
            fetch_slow_queries,
            fetch_single_query,
            fetch_slow_queries_batch,
            fetch_fastest_queries,
            # TPOT & KPI
            get_token_velocity,
            analyze_request_queuing,
            check_kpi_compliance,
            analyze_thinking_overhead,
            detect_compute_inefficiency,
            # GenerationConfig analysis
            get_generation_config_comparison,
            analyze_config_correlation,
            get_config_outliers,
            # Hourly model analysis
            get_hourly_model_distribution,
            get_hourly_model_latency_heatmap,
            # Configuration
            get_analysis_config,
            get_analysis_config,
            get_analysis_metadata,
            verify_data_access,
            save_analysis_report,
            get_subagent_tool_usage
        ],
        generate_content_config=CONTENT_CONFIG,
        output_key=KEY_DOC_OUTPUT,
        after_model_callback=accumulate_investigator_output
    )

    # 4. Critique
    critique = LlmAgent(
        name=f"critique_{safe_name}",
        model=MODEL,
        description="Hostile reviewer of findings",
        instruction=CRITIQUE_PROMPT + f"\nYour Input is '{{ {KEY_DOC_OUTPUT} ? }}'",
        tools=[get_analysis_metadata],
        output_schema=InvestigationFeedback,
        disallow_transfer_to_parent=True,
        disallow_transfer_to_peers=True,
        output_key=KEY_FEEDBACK
    )

    # 5. Escalator
    escalator = InvestigationEscalationChecker(name=f"escalator_{safe_name}", feedback_key=KEY_FEEDBACK)

    loop = LoopAgent(
        name=f"loop_{safe_name}",
        max_iterations=3,
        sub_agents=[investigator, critique, escalator]
    )

    # 6. Section Writer
    writer = LlmAgent(
        name=f"writer_{safe_name}",
        model=MODEL,
        description="Writes final section markdown",
        instruction=SECTION_WRITER_PROMPT + f"\nInput: Findings from {{ {KEY_DOC_OUTPUT} ? }}",
        generate_content_config=CONTENT_CONFIG,
        output_key=KEY_FINAL_SECTION
    )

    return SequentialAgent(
        name=f"TEAM_{safe_name}",
        sub_agents=[primer, strategist, loop, writer]
    )

# =========================================
# MAIN AGENTS
# =========================================

final_report_assembler = LlmAgent(
    name="final_report_assembler",
    model=MODEL,
    description="Combines all individual report sections into the final master document.",
    instruction=FINAL_REPORT_ASSEMBLER_PROMPT,
    generate_content_config=CONTENT_CONFIG,
    tools=[get_analysis_metadata, get_tool_usage_report],
    output_key="FINAL_REPORT_MARKDOWN"
)

markdown_corrector = LlmAgent(
    name="markdown_corrector",
    model=MODEL,
    description="Fixes markdown formatting and table issues in the final report.",
    instruction=MARKDOWN_CORRECTOR_PROMPT + "\nInput: content from 'FINAL_REPORT_MARKDOWN'",
    generate_content_config=CONTENT_CONFIG,
    output_key="FINAL_REPORT_MARKDOWN_CORRECTED"
)

report_saver = LlmAgent(
    name="report_saver",
    model=MODEL,
    description="Saves the generated report to a timestamped file.",
    instruction=REPORT_SAVER_PROMPT + "\nInput: content from 'FINAL_REPORT_MARKDOWN_CORRECTED'",
    generate_content_config=types.GenerateContentConfig(temperature=0),
    tools=[save_analysis_report]
)

# Parallel Execution Engine
complete_report_generator = SequentialAgent(
    name="complete_report_generator",
    description="PARALLEL EXECUTION: Launches all analysis teams at once, then assembles the final report.",
    sub_agents=[
        # SCATTER: Run all teams in parallel
        ParallelAgent(
            name="investigation_swarm",
            sub_agents=[build_dimension_team(dim) for dim in CURRENT_DIMENSION_LIST]
        ),
        # GATHER: Combine results
        final_report_assembler
    ],
)

# Wrapper that handles the entire Generate -> Save flow
report_orchestrator = SequentialAgent(
    name="report_orchestrator",
    description="Orchestrator that generates the report and then automatically saves it.",
    sub_agents=[
        complete_report_generator,
        markdown_corrector,
        report_saver
    ]
)

# Root Entry Point
parallel_latency_analyzer = LlmAgent(
    name="parallel_latency_analyzer",
    model=MODEL,
    description="Advanced Latency Consultant. Can run parallel deep-dive analysis across multiple dimensions.",
    instruction=ROOT_AGENT_PROMPT,
    generate_content_config=types.GenerateContentConfig(temperature=0),
    tools=[
        trigger_latency_parallel_report, 
        process_latency_question, 
        save_analysis_report, 
        read_report_content, 
        has_report_content,
        get_analysis_config,
        get_query_details,
        get_request_details,
        get_subagent_tool_usage
    ],
    sub_agents=[report_orchestrator],
)

root_agent = parallel_latency_analyzer

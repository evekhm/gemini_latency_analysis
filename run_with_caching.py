#!/usr/bin/env python3
"""
Autonomous Latency Analysis Runner with ADK Context Caching

This script runs the parallel latency analyzer with proper ADK Context Caching
configured at the App level, following ADK best practices.
"""
import agentops
import asyncio
import logging
import os
import sys
import json
from pathlib import Path

from dotenv import load_dotenv
from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.apps.app import App
from google.adk.cli.utils import logs
from google.adk.runners import InMemoryRunner
from google.genai import types
from opentelemetry import trace
from agents.parallel_latency_analyzer.telemetry import init_tracer

# Add agents directory to path
agents_dir = Path(__file__).parent / "agents"
sys.path.insert(0, str(agents_dir))

# Import the agent
# Import moved to inside run_autonomous_analysis to allow logger setup first
# from parallel_latency_analyzer.agent import parallel_latency_analyzer

# Load environment variables
load_dotenv()

# Configuration
APP_NAME = "parallel_latency_analyzer_cached"
USER_ID = "latency_analyst"

# ADK Context Caching Configuration
# These settings optimize for the parallel latency analyzer's usage pattern
CACHE_CONFIG = ContextCacheConfig(
    min_tokens=4096,      # Cache requests with 4K+ tokens (typical for our analysis)
    ttl_seconds=1800,     # Cache for 30 minutes (duration of analysis run)
    cache_intervals=10,   # Allow up to 10 uses before refresh
)


async def run_autonomous_analysis(replay_file_path: str = None):
    """Run autonomous latency analysis with context caching enabled."""
    
    print("=" * 80)
    print("  Parallel Autonomous Latency Analysis")
    print("  With ADK Context Caching Enabled")
    if replay_file_path:
        print(f"  Replay File: {replay_file_path}")
    print("=" * 80)
    print()
    
    # Verify environment
    project_id = os.getenv('PROJECT_ID')
    dataset_id = os.getenv('DATASET_ID')
    table_id = os.getenv('AGENT_TABLE_ID', 'gemini_logs')
    
    if not project_id or not dataset_id:
        print("❌ Error: PROJECT_ID and DATASET_ID must be set in .env")
        sys.exit(1)
    
    print(f"✓ Project: {project_id}")
    print(f"✓ Dataset: {dataset_id}")
    print(f"✓ Table(s): {table_id}")
    print()
    
    # Load replay file if provided
    initial_state = {}
    queries = ["Run autonomous latency analysis for the configured time period."]
    
    if replay_file_path:
        try:
            with open(replay_file_path, 'r') as f:
                replay_data = json.load(f)
                
            # Merge state AND config into initial state
            # adk run --replay puts the root keys into session state
            initial_state = replay_data.get("state", {})
            if "config" in replay_data:
                initial_state["config"] = replay_data["config"]
                
            # Override queries if present
            if "queries" in replay_data and replay_data["queries"]:
                queries = replay_data["queries"]
                
            print(f"✓ Loaded replay configuration from {replay_file_path}")
            
        except Exception as e:
            print(f"⚠️ Warning: Failed to load replay file: {e}")
            print("  Falling back to default configuration.")

    print()
    print("📊 Context Caching Configuration:")
    print(f"   Min Tokens: {CACHE_CONFIG.min_tokens}")
    print(f"   TTL: {CACHE_CONFIG.ttl_seconds}s ({CACHE_CONFIG.ttl_seconds // 60} minutes)")
    print(f"   Cache Intervals: {CACHE_CONFIG.cache_intervals}")
    print()

    AGENTOPS_API_KEY = os.getenv("AGENTOPS_API_KEY")
    if AGENTOPS_API_KEY:
        agentops.init(
            api_key=AGENTOPS_API_KEY,
            default_tags=['google adk'],
            trace_name="agent-analyzer-trace"
        )

    # Create App with context caching
    from parallel_latency_analyzer.agent import parallel_latency_analyzer
    app = App(
        name=APP_NAME,
        root_agent=parallel_latency_analyzer,
        context_cache_config=CACHE_CONFIG,
    )
    
    # Create runner
    runner = InMemoryRunner(app=app)
    
    # Create session
    session = await runner.session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID
    )
    
    # Initialize session state from replay data
    if initial_state:
        # We need to manually update the session state
        # InMemorySessionService usually allows direct access or update
        # For simplicity, we loop and set items, or update if supported
        for k, v in initial_state.items():
            session.state[k] = v
        print(f"✓ Initialized session state with {len(initial_state)} keys")
    
    print(f"✓ Session created: {session.id}")
    print()
    print("🚀 Starting autonomous analysis...")
    print()
    
    
    
    # Run the analysis for each query
    import time
    start_time = time.time()
    
    # Track token usage
    total_input_tokens = 0
    total_output_tokens = 0
    
    # Track tool usage
    from collections import Counter
    tool_usage = Counter()
    
    try:
        for query in queries:
            print(f"➤ Sending query: {query}")
            
            # Wrap query in Content object as expected by ADK runner
            message_content = types.Content(
                role="user",
                parts=[types.Part.from_text(text=query)]
            )
            
            async for event in runner.run_async(
                user_id=USER_ID,
                session_id=session.id,
                new_message=message_content
            ):
                # Print agent responses
                if hasattr(event, 'content') and event.content:
                    print(f"[{event.author}] {event.content}")
                
                # Accumulate usage from event metadata
                if hasattr(event, 'usage_metadata') and event.usage_metadata:
                    usage = event.usage_metadata
                    # Handle object or dict access
                    if hasattr(usage, 'prompt_token_count'):
                        total_input_tokens += (usage.prompt_token_count or 0)
                    if hasattr(usage, 'candidates_token_count'):
                        total_output_tokens += (usage.candidates_token_count or 0)
                    # Note: We rely on the event stream potentially sending usage 
                    # for each turn. This might count duplicates if multiple events share usage.
                    # Typically usage is sent once per generation completion.

                # Track tool calls from model output items
                if hasattr(event, 'content') and event.content and hasattr(event.content, 'parts'):
                    for part in event.content.parts:
                        if hasattr(part, 'function_call') and part.function_call:
                             tool_name = part.function_call.name
                             tool_usage[tool_name] += 1
                    
            print("-" * 40)
    
    except KeyboardInterrupt:
        print("\n⚠️  Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        end_time = time.time()
        elapsed_time = end_time - start_time
        total_tokens = total_input_tokens + total_output_tokens
        
        print()
        print("=" * 80)
        print("  Analysis Complete!")
        print("=" * 80)
        
        print("\n📊 Session Summary")
        print(f"   ⏱️  Elapsed Time: {elapsed_time:.2f}s ({elapsed_time/60:.2f}m)")
        print(f"   🔢 Total Tokens: {total_tokens:,}")
        print(f"      - Input:  {total_input_tokens:,}")
        print(f"      - Output: {total_output_tokens:,}")

            
    print("  Check reports/ for the analysis report")
    print("=" * 80)

    # NEW: Append Execution Summary to the generated report
    try:
        # Find the latest report file in reports/ directory based on modification time
        reports_dir = Path(__file__).parent / "reports"
        if reports_dir.exists():
            # Get all md files
            report_files = list(reports_dir.glob("*.md"))
            if report_files:
                # Sort by modification time, newest first
                latest_report = max(report_files, key=lambda p: p.stat().st_mtime)
                
                # Double check it was modified recently (e.g. within the last minute)
                # to avoid appending to an old report if this run failed to generate one.
                # But for now, we'll just trust it's the right one or check if it matches our timestamp logic if possible.
                # Actually, simpler: just append to the *newest* file.
                
                summary_md = f"""
## Agent Execution Summary

| Metric | Value |
| :--- | :--- |
| **Elapsed Time** | {elapsed_time:.2f}s ({elapsed_time/60:.2f}m) |
| **Total Tokens** | {total_tokens:,} |
| **Input Tokens** | {total_input_tokens:,} |
| **Output Tokens** | {total_output_tokens:,} |

### Tool Usage Statistics

| Tool Name | Count |
| :--- | :--- |
"""
                # Add rows for tool usage
                for tool, count in tool_usage.most_common():
                    summary_md += f"| `{tool}` | {count} |\n"
                    
                summary_md += f"\n*Generated by: {APP_NAME}*\n"

                with open(latest_report, "a") as f:
                    f.write(summary_md)
                
                print(f"✓ Appended execution summary to: {latest_report.name}")
            else:
                print("⚠️ No report files found to append summary.")
        else:
             print("⚠️ Reports directory not found.")
             
    except Exception as e:
        print(f"⚠️ Failed to append summary to report: {e}")



def main():
    """Main entry point."""
    # Setup logging
    logs.setup_adk_logger(logging.INFO)

    # Setup OpenTelemetry Tracing
    init_tracer()
    
    # Manual Logger Setup to ensure file creation
    try:
        import tempfile
        from datetime import datetime
        
        # Create log directory
        log_dir = Path(tempfile.gettempdir()) / "agents_log"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"agent.{timestamp}.log"
        
        # Create FileHandler
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Define TraceIdFilter
        class TraceIdFilter(logging.Filter):
            def filter(self, record):
                span = trace.get_current_span()
                if span == trace.INVALID_SPAN:
                    return True
                ctx = span.get_span_context()
                if ctx == trace.INVALID_SPAN_CONTEXT:
                    return True
                
                project_id = os.getenv('PROJECT_ID')
                if project_id:
                    # Inject trace info into the record for Cloud Logging
                    record.trace = f"projects/{project_id}/traces/{trace.format_trace_id(ctx.trace_id)}"
                    record.span_id = trace.format_span_id(ctx.span_id)
                    record.trace_sampled = ctx.trace_flags.sampled
                return True

        # Attach to root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)
        
        # Add Trace Filter to propagate trace context to logs
        trace_filter = TraceIdFilter()
        root_logger.addFilter(trace_filter)
        
        # Also attach to 'adk' and 'google' loggers to be safe
        logging.getLogger('adk').addHandler(file_handler)
        logging.getLogger('google').addHandler(file_handler)
        
        # Ensure 'adk' logger also gets the filter if it doesn't propagate
        logging.getLogger('adk').addFilter(trace_filter)
        
        print(f"✓ Created log file: {log_file}")
        
        if log_file:
            symlink_path = Path("latest_agent.log")
            
            # Remove existing symlink or file
            if symlink_path.exists() or symlink_path.is_symlink():
                symlink_path.unlink()
            
            # Create new symlink
            symlink_path.symlink_to(log_file)
            print(f"✓ Created symlink: latest_agent.log -> {log_file}")
            print(f"  Tip: Run 'tail -f latest_agent.log' to monitor agent progress")
             
    except Exception as e:
        print(f"⚠️ Warning: Failed to setup manual logging: {e}")
    
    # Parse args (simple manual check for now or just hardcode the expected path)
    # The shell script is expected to direct us or we default relative
    replay_path = None
    if len(sys.argv) > 1:
        replay_path = sys.argv[1]
    
    # Run the analysis
    asyncio.run(run_autonomous_analysis(replay_path))


if __name__ == "__main__":
    main()

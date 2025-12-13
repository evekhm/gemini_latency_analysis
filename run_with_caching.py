#!/usr/bin/env python3
"""
Autonomous Latency Analysis Runner with ADK Context Caching

This script runs the parallel latency analyzer with proper ADK Context Caching
configured at the App level, following ADK best practices.
"""

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
            print("-" * 40)
    
    except KeyboardInterrupt:
        print("\n⚠️  Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print()
    print("=" * 80)
    print("  Analysis Complete!")
    print("  Check reports/ for the analysis report")
    print("=" * 80)


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
        
        # Attach to root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)
        
        # Also attach to 'adk' and 'google' loggers to be safe
        logging.getLogger('adk').addHandler(file_handler)
        logging.getLogger('google').addHandler(file_handler)
        
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

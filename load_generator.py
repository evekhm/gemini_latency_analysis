
import asyncio
import os
import logging
import argparse
import time
import json
from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.genai import types
from google.adk import Runner
from google.adk.models import LlmResponse
from google.adk.sessions import InMemorySessionService

__dir__ = os.path.dirname(__file__)
load_dotenv(dotenv_path=os.path.join(__dir__, "../../.env"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Load environment variables
load_dotenv()

PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = os.getenv("LOCATION", "us-central1")
MODEL_ID = os.getenv("MODEL")

assert PROJECT_ID, "PROJECT_ID is not set"

# ADK/GenAI SDK often looks for GOOGLE_CLOUD_PROJECT
if "GOOGLE_CLOUD_PROJECT" not in os.environ:
    os.environ["GOOGLE_CLOUD_PROJECT"] = PROJECT_ID
if "GOOGLE_CLOUD_LOCATION" not in os.environ:
    os.environ["GOOGLE_CLOUD_LOCATION"] = LOCATION

assert MODEL_ID, "Model must be set!"

def create_load_generator_agent(name: str, model: str = "gemini-2.0-flash-exp") -> LlmAgent:
    return LlmAgent(
        name=name,
        model=model,
        description="A simple load generator agent for latency analysis.",
        instruction="You are a helpful assistant. Respond to the user's prompt directly.",
        generate_content_config=types.GenerateContentConfig(
            temperature=0.7,
            max_output_tokens=8192,
            labels={"adk_agent_name": name} # Explicitly set label for BigQuery tracking
        )
    )

def generate_large_prompt(token_count_approx: int) -> str:
    """Generates a large prompt by repeating a phrase."""
    base_phrase = "The quick brown fox jumps over the lazy dog. "
    repeats = max(1, token_count_approx // 10)
    return base_phrase * repeats + "\n\nSummarize the above in one word."

async def send_request(agent_name: str, prompt: str, max_output_tokens: int, label: str, streaming: bool = False):
    """Sends a single request using an ADK Agent and logs latency (TTFT and E2E)."""
    start_time = time.time()
    ttft = 0
    first_token_time = None
    text_content = ""
    
    # Check if we should use agent or direct client
    # We can pass a flag or check config. For now, let's assume if agent_name is "direct", we use direct client.
    # Or better, let's add a parameter to send_request.
    # But send_request signature is fixed for now.
    # Let's handle it inside send_request.
    
    if agent_name == "direct":
        # Direct client usage
        # We need to instantiate a client here or pass it in.
        # Since we removed client from args, we instantiate it locally or globally.
        # For efficiency, we should probably reuse it, but for load gen, creating it is fine or we can make it global.
        from google.genai import Client
        client = Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
        
        try:
            logging.info(f"[{label}] Sending request via Direct Client...")
            if streaming:
                async for chunk in await client.aio.models.generate_content_stream(
                    model=MODEL_ID,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.7,
                        max_output_tokens=max_output_tokens,
                        labels={"adk_agent_name": "direct_client"}
                    )
                ):
                    if first_token_time is None:
                        first_token_time = time.time()
                        ttft = first_token_time - start_time
                    if chunk.text:
                        text_content += chunk.text
                end_time = time.time()
                e2e_latency = end_time - start_time
            else:
                response = await client.aio.models.generate_content(
                    model=MODEL_ID,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.7,
                        max_output_tokens=max_output_tokens,
                        labels={"adk_agent_name": "direct_client"}
                    )
                )
                end_time = time.time()
                e2e_latency = end_time - start_time
                text_content = response.text if response.text else ""
                
            text_len = len(text_content)
            if first_token_time is None:
                ttft = e2e_latency
            logging.info(f"[{label}] Success! TTFT: {ttft:.2f}s | E2E: {e2e_latency:.2f}s | Output len: {text_len}")
            return e2e_latency, text_content
        except Exception as e:
            logging.error(f"[{label}] Failed: {e}")
            return None, None

    # Create a fresh agent for this request
    agent = create_load_generator_agent(name=agent_name, model=MODEL_ID)
    
    # We need a Runner to execute the agent
    # Using InMemorySessionService for lightweight execution
    runner = Runner(agent=agent, app_name="load_generator", session_service=InMemorySessionService())
    
    try:
        logging.info(f"[{label}] Sending request via Agent '{agent_name}'...")
        
        session_id = f"session_{label}"
        await runner.session_service.create_session(session_id=session_id, user_id="load_test_user", app_name="load_generator")
        
        # We need to collect events to measure TTFT
        async for event in runner.run_async(
            user_id="load_test_user",
            session_id=session_id,
            new_message=types.Content(role="user", parts=[types.Part(text=prompt)])
        ):
            if isinstance(event, LlmResponse):
                if first_token_time is None:
                    first_token_time = time.time()
                    ttft = first_token_time - start_time
                
                if event.content:
                    # Depending on the event structure, content might be a string or object
                    # ModelResponse usually has 'content' which is the text chunk or full text
                    # We'll append it if it's a string
                    if isinstance(event.content, str):
                        text_content += event.content
                    elif hasattr(event.content, 'parts'):
                         for part in event.content.parts:
                             if part.text:
                                 text_content += part.text

        end_time = time.time()
        e2e_latency = end_time - start_time
        text_len = len(text_content)
        
        # If TTFT wasn't captured (e.g. non-streaming or single response), it's same as E2E
        if first_token_time is None:
            ttft = e2e_latency

        logging.info(f"[{label}] Success! TTFT: {ttft:.2f}s | E2E: {e2e_latency:.2f}s | Output len: {text_len}")

        if not text_content:
             logging.warning(f"[{label}] Response text is empty.")
             
        return e2e_latency, text_content
    except Exception as e:
        logging.error(f"[{label}] Failed: {e}")
        return None, None

async def run_scenario(name: str, config: dict, override_count: int = None, agent_name: str = "load_generator"):
    """Runs a specific scenario based on config."""
    count = override_count or config.get("count", 1)
    concurrency = config.get("concurrency", 1)
    description = config.get("description", "")
    streaming = config.get("streaming", False) # Default to False
    
    # Use the agent_name from config if available, otherwise default
    scenario_agent_name = config.get("agent_name", agent_name)
    
    logging.info(f"--- Starting Scenario: {name} (Agent: {scenario_agent_name}, {count} requests, concurrency={concurrency}) ---")
    logging.info(f"Description: {description}")
    
    # Prepare prompt
    if config.get("prompt_type") == "generated":
        prompt = generate_large_prompt(config.get("prompt_length", 1000))
    else:
        prompt = config.get("prompt", "Hello")
        
    max_output_tokens = config.get("max_output_tokens", 5000)
    
    tasks = []
    latencies = []
    
    if concurrency > 1:
        # Concurrent execution
        for i in range(count):
            tasks.append(send_request(scenario_agent_name, prompt, max_output_tokens, label=f"{name}-{i+1}", streaming=streaming))
        
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        latencies = [r[0] for r in results if r[0] is not None]
        logging.info(f"Scenario {name} Completed in {total_time:.2f}s")
        
    else:
        # Sequential execution
        for i in range(count):
            lat, _ = await send_request(scenario_agent_name, prompt, max_output_tokens, label=f"{name}-{i+1}", streaming=streaming)
            if lat: latencies.append(lat)
            
    if latencies:
        avg = sum(latencies) / len(latencies)
        logging.info(f"[{name}] Average Latency: {avg:.2f}s")


async def send_hello_world(agent_name: str = "load_generator"):
    """Runs a simple Hello World check."""
    logging.info(f"--- Running Default: Hello World Check (Agent: {agent_name}) ---")
    latency, text = await send_request(agent_name, "Hello World", 1000, "HelloWorld", streaming=False)
    if text:
        print(f"\nModel Answer: {text}\n")

async def main():
    print(f"Load Generator using Project ID: {PROJECT_ID}")
    parser = argparse.ArgumentParser(description="Load Generator for Latency Analysis")
    parser.add_argument("scenario", nargs="?", help="Scenario to run (key in load_scenarios.json) or 'all'")
    parser.add_argument("--count", type=int, default=None, help="Override number of requests")
    parser.add_argument("--config", default="load_scenarios.json", help="Path to config file")
    parser.add_argument("--agent-name", default="load_generator", help="Default agent name if not in config")
    args = parser.parse_args()

    if not PROJECT_ID:
        logging.error("PROJECT_ID environment variable not set.")
        return

    # Load config
    try:
        with open(args.config, 'r') as f:
            full_config = json.load(f)
            scenarios = full_config.get("scenarios", {})
    except Exception as e:
        logging.error(f"Failed to load config file {args.config}: {e}")
        return

    if not args.scenario:
        await send_hello_world(args.agent_name)
        return

    if args.scenario == "all":
        for name, config in scenarios.items():
            await run_scenario(name, config, args.count, args.agent_name)
    elif args.scenario in scenarios:
        await run_scenario(args.scenario, scenarios[args.scenario], args.count, args.agent_name)
    else:
        logging.error(f"Scenario '{args.scenario}' not found in config. Available: {list(scenarios.keys())}")

if __name__ == "__main__":
    asyncio.run(main())




"""
Test basic ADK agent functionality.
Verifies that agent instantiation and execution work correctly.
"""

import asyncio
import os
from google.adk import Agent
from google.adk.runners import InMemoryRunner
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# Load configuration from .env
PROJECT_ID = os.getenv("PROJECT_ID")
REGION = os.getenv("REGION")
MODEL = os.getenv("MODEL", "gemini-2.0-flash")

# Set Google SDK environment variables from .env values
# The SDK expects these specific variable names
if PROJECT_ID:
    os.environ["GOOGLE_CLOUD_PROJECT"] = PROJECT_ID
if REGION:
    os.environ["GOOGLE_CLOUD_LOCATION"] = REGION

async def test_agent():
    """Test basic agent execution with ADK."""
    
    # Skip test if credentials aren't configured
    if not PROJECT_ID or not REGION:
        print("⚠️  Skipping test - Vertex AI credentials not configured")
        print(f"   Found PROJECT_ID: {PROJECT_ID}")
        print(f"   Found REGION: {REGION}")
        print("   Please check .env file")
        print("✅ Test SKIPPED (not a failure)")
        return 0
    
    # Use the configured model from .env
    app_name = "test_agent_app"
    user_id = "test_user"
    
    print(f"Testing agent with model: {MODEL}")
    print(f"Using project: {PROJECT_ID}, region: {REGION}")
    
    # 2. Define the agent
    agent = Agent(
        name="test_agent",
        model=MODEL,  # Use MODEL from .env
        instruction="You are a helpful assistant. Reply with 'Hello' when greeted."
    )
    
    # 3. Create runner with in-memory session
    runner = InMemoryRunner(
        agent=agent,
        app_name=app_name
    )
    
    # 4. Create session
    session = await runner.session_service.create_session(
        app_name=app_name,
        user_id=user_id
    )
    
    print(f"Running agent {agent.name}...")
    
    # 5. Format user message
    user_message = types.Content(
        role="user",
        parts=[types.Part.from_text(text="Hi")]
    )
    
    # 6. Run and collect response
    response_received = False
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session.id,
        new_message=user_message
    ):
        if event.content and event.content.parts:
            response_text = event.content.parts[0].text
            if response_text:
                print(f"✓ Agent responded: {response_text[:50]}...")
                response_received = True
    
    # 7. Verify we got a response
    if response_received:
        print("✅ Test PASSED - Agent executed successfully")
        return 0
    else:
        print("❌ Test FAILED - No response from agent")
        return 1

if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(test_agent())
    sys.exit(exit_code)
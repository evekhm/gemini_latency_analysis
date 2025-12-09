
import asyncio
from google.adk.agents import LlmAgent, InvocationContext
from google.adk.models import LlmResponse

async def test_run():
    agent = LlmAgent(name="test", model="gemini-pro")
    
    # Try creating a context
    # InvocationContext usually takes the agent and input
    # Let's inspect InvocationContext again or try to instantiate it
    try:
        ctx = InvocationContext(agent=agent, input="Hello")
        print("Created context with input string")
    except Exception as e:
        print(f"Failed to create context: {e}")
        
    # Try running with context
    try:
        async for event in agent.run_async(ctx):
            print(event)
    except Exception as e:
        print(f"Failed to run with context: {e}")

if __name__ == "__main__":
    asyncio.run(test_run())

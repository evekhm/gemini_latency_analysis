import asyncio
import uuid
from google.adk.agents import LlmAgent, InvocationContext
from google.adk.sessions import InMemorySessionService, Session
from google.adk.models import LlmResponse

async def test_run():
    agent = LlmAgent(name="test", model="gemini-pro")
    
    # Use real objects for strict Pydantic validation
    session_service = InMemorySessionService()
    session = Session(id=str(uuid.uuid4()), app_name="test_app", user_id="test_user")
    
    # Try creating a context
    try:
        ctx = InvocationContext(
            agent=agent,
            session_service=session_service,
            invocation_id=str(uuid.uuid4()),
            session=session
        )
        print("Created context with required fields")
    except Exception as e:
        print(f"Failed to create context: {e}")
        return

    try:
        print("✅ Context creation successful")
        
    except Exception as e:
        print(f"Failed to run with context: {e}")

if __name__ == "__main__":
    asyncio.run(test_run())

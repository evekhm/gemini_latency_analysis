import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.latency_analyzer.query_extractor import extract_user_query

# Real case from the logs - simplified structure
real_case = {
  "contents": [{
    "role": "user",
    "parts": [{
      "text": """You are a helpful conversational agent for the United Healthcare Group (UHG)...
[MASSIVE SYSTEM PROMPT WITH INSTRUCTIONS AND EXAMPLES]
</Context>



Provide the JSON response for the round number 1 for the question I injured my back. Is massage therapy covered?.

If previous conversation is not empty, analyze the previous question and the answer.
Reason about how current question is connected to the previous conversation."""
    }]
  }]
}

print("Testing with real case structure...")
result = extract_user_query(real_case)
print(f"Result: '{result}'")
print(f"Length: {len(result)}")

expected_substring = "I injured my back"
if expected_substring in result and "You are a helpful" not in result:
    print("✅ Test PASSED - Extracted user query correctly")
else:
    print(f"❌ Test FAILED - Should contain '{expected_substring}' without system prompt")
    print(f"   Contains user query: {expected_substring in result}")
    print(f"   Contains system prompt: {'You are a helpful' in result}")

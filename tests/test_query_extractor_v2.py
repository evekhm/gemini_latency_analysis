import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.latency_analyzer.query_extractor import extract_user_query
import json

# Case 1: Multi-message (User's first example)
example_1 = {
  "contents": [
    {"parts":[{"text":"hi"}],"role":"user"},
    {"parts":[{"text":"For context: ..."}],"role":"user"},
    {"parts":[{"text":"A"}],"role":"user"}
  ]
}

# Case 2: Massive Context Stuffing (Found in logs)
example_2 = {
  "contents": [
    {
      "role": "user",
      "parts": [{
        "text": "You are a helpful agent... <Context> [MASSIVE DATA] </Context>\n\nProvide the JSON response for the round number 1 for the question I injured my back. Is massage therapy covered?.\n\nIf previous conversation is not empty..."
      }]
    }
  ]
}

print("Testing extraction logic...")

# Test 1
res1 = extract_user_query(example_1)
print(f"Case 1 Result: '{res1}'")
if res1 == "hi | A":
    print("✅ Case 1 PASSED")
else:
    print(f"❌ Case 1 FAILED. Got: '{res1}'")

# Test 2
res2 = extract_user_query(example_2)
print(f"Case 2 Result: '{res2}'")
# We want to see the question, not the system prompt
if "injured my back" in res2 and "helpful agent" not in res2:
    print("✅ Case 2 PASSED")
else:
    print(f"❌ Case 2 FAILED. Got: '{res2}'")

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.parallel_latency_analyzer.query_extractor import extract_user_query
import json

# User provided example structure
example_request = {
  "contents": [
    {"parts":[{"text":"hi"}],"role":"user"},
    {"parts":[{"text":"For context: [legal_consultant] called tool..."}],"role":"user"},
    {"parts":[{"text":"For context: [legal_consultant] tool returned..."}],"role":"user"},
    {"parts":[{"text":"For context: [legal_consultant] said: ..."}],"role":"user"},
    {"parts":[{"text":"A"}],"role":"user"},
    {"parts":[{"text":"For context: [legal_consultant] said: Starting..."}],"role":"user"},
    {"parts":[{"text":"For context: [legal_consultant] tool returned..."}],"role":"user"}
  ]
}

print("Testing extraction logic...")
result = extract_user_query(example_request)
print(f"Result: '{result}'")

expected = "hi | A"
if result == expected:
    print("✅ Test PASSED")
else:
    print(f"❌ Test FAILED. Expected '{expected}', got '{result}'")

# Configuration & Agent Tests

This directory contains tests to verify configuration handling, query extraction, and ADK agent integration.

## Test Files

### Configuration Tests

- **`test_config.py`** - Tests a single configuration file
  - Verifies all expected fields are present and accessible
  - Checks nested KPI fields (mean_latency_target, p95_latency_target)
  - Simulates agent usage pattern with config parsing

- **`test_all_configs.py`** - Tests all configuration files in the project
  - Auto-discovers all `*analysis*.json` and `*kpi*.json` files
  - Validates structure and required fields for each
  - Reports on individual file validation

### Query Extraction Tests

- **`test_query_extractor.py`** - Basic query extraction tests
  - Validates extraction of user queries from request JSON
  - Tests filtering out "For context:" messages
  - Ensures multi-message queries are properly joined with " | "

- **`test_query_extractor_v2.py`** - Extended query extraction tests
  - Tests multiple edge cases and scenarios
  - Validates extraction from complex nested structures
  - Checks handling of system prompts vs user queries
  - Tests massive context stuffing scenarios

- **`test_real_case.py`** - Real log structure tests
  - Uses actual request structures from production logs
  - Validates query extraction in production scenarios
  - Tests extraction from UHG-style request format

### ADK Integration Tests

- **`test_agent_run.py`** - ADK agent execution test
  - Tests basic agent instantiation and execution
  - Uses `Agent` + `InMemoryRunner` pattern
  - Validates agent can call Gemini API and receive responses
  - Uses .env configuration (PROJECT_ID, REGION, MODEL)
  - **Note**: Requires Vertex AI credentials to run

- **`test_context.py`** - ADK context API exploration
  - Tests InvocationContext creation (expected to show errors)
  - Kept for API exploration and reference

### Test Infrastructure

- **`run_tests.py`** - Test runner script
  - Auto-discovers all `test_*.py` files in this directory
  - Runs each test and captures output
  - Reports overall pass/fail status with summary

## Running Tests

### Run all tests (recommended)
```bash
# From project root
./run_tests.sh

# Or directly with Python
python tests/run_tests.py
```

### Run individual tests
```bash
# From project root
python tests/test_config.py
python tests/test_all_configs.py
python tests/test_query_extractor.py
python tests/test_agent_run.py
```

## Expected Output

All 7 tests should pass:

```
================================================================================
TEST SUMMARY
================================================================================
✅ PASS: test_agent_run.py
✅ PASS: test_all_configs.py
✅ PASS: test_config.py
✅ PASS: test_context.py
✅ PASS: test_query_extractor.py
✅ PASS: test_query_extractor_v2.py
✅ PASS: test_real_case.py

Results: 7/7 tests passed

🎉 All tests passed!
```

## What These Tests Verify

### Configuration Structure
- Files have proper `config` section
- Required field `analysis_scope` is present ("standard" | "autonomous" | "deep_research")
- Optional fields are properly structured

### Configuration Fields
- `time_period_days`: Time range (e.g., "24h", "90d", "last 27 days")
- `analysis_scope`: Workflow type
- `kpis`: KPI targets object with sub-fields
  - `mean_latency_target`: Target mean latency in seconds
  - `p95_latency_target`: Target P95 latency in seconds
- `num_slowest_queries`: Number of slow queries to analyze
- `agent_name`: Specific agent filter or null for all agents

### Agent Functionality
- Config can be read via `get_analysis_config()`
- JSON parsing works correctly
- All fields are accessible with correct types
- Query extraction from complex JSON works reliably
- ADK agent can execute and receive responses from Gemini API

## Environment Requirements

### For Configuration Tests
- No special requirements (always work)

### For ADK Agent Test (`test_agent_run.py`)
Requires `.env` file with:
```
PROJECT_ID=your-gcp-project
REGION=us-central1
MODEL=gemini-2.5-pro
```

The test will skip gracefully if credentials aren't configured.

## Adding New Tests

To add a new test:

1. Create `test_<name>.py` in this directory
2. Ensure it returns exit code 0 for success, 1 for failure
3. Add docstring explaining what the test validates
4. Run `python tests/run_tests.py` to verify auto-discovery

The test runner automatically discovers all `test_*.py` files.

## Test Organization

Tests follow the user's global rules:
- All test files in `tests/` directory
- Use sys.path modification to import from parent:
  ```python
  import sys
  import os
  sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
  ```

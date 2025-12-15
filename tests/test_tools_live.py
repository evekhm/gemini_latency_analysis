
import os
import json
import logging
import unittest
import sys
from dotenv import load_dotenv

# Add project root to sys.path to allow importing agents
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.parallel_latency_analyzer.utils import (
    get_overall_statistics,
    get_model_comparison,
    get_agent_comparison,
    fetch_slow_queries_batch,
    fetch_fastest_queries,
    check_kpi_compliance,
    cluster_slow_queries,
    get_token_velocity,
    analyze_thinking_overhead,
    detect_compute_inefficiency,
    get_generation_config_comparison,
    analyze_config_correlation,
    get_config_outliers,
    get_agent_model_matrix,
    get_query_details
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)

# Global test configuration
TIME_RANGE = "100d"
NUM_QUERIES = 5

class TestToolsLive(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Verify environment
        if not os.getenv('PROJECT_ID'):
            raise EnvironmentError("PROJECT_ID is missing")
        if not os.getenv('DATASET_ID'):
            raise EnvironmentError("DATASET_ID is missing")

    def _check_for_error(self, result, tool_name):
        """Parses result and asserts no error key."""
        data = None
        if isinstance(result, str):
            try:
                data = json.loads(result)
            except json.JSONDecodeError:
                # If it's a plain string (not JSON), we might accept it if it's not an error message
                # But most tools return JSON.
                if result.startswith("Error"):
                    self.fail(f"{tool_name} returned error string: {result}")
                return result # Accept raw string if it's not an error
        else:
            data = result
        
        self.assertIsNotNone(data, f"{tool_name} returned None or invalid data")
        
        if isinstance(data, dict):
            if "error" in data:
                self.fail(f"{tool_name} returned error: {data['error']}")
        
        return data

    def test_get_overall_statistics(self):
        tool_name = "get_overall_statistics"
        logging.info(f"Testing {tool_name}...")
        result = get_overall_statistics(time_range=TIME_RANGE)
        data = self._check_for_error(result, tool_name)
        # Expecting list or dict with stats
        self.assertTrue(isinstance(data, list) or "overall_stats" in data or "total_requests" in data or len(data) > 0)

    def test_get_model_comparison(self):
        tool_name = "get_model_comparison"
        logging.info(f"Testing {tool_name}...")
        result = get_model_comparison(time_range=TIME_RANGE)
        data = self._check_for_error(result, tool_name)
        self.assertIn("models", data)

    def test_get_agent_comparison(self):
        tool_name = "get_agent_comparison"
        logging.info(f"Testing {tool_name}...")
        result = get_agent_comparison(time_range=TIME_RANGE)
        data = self._check_for_error(result, tool_name)
        self.assertIn("agents", data)

    def test_fetch_slow_queries_batch(self):
        tool_name = "fetch_slow_queries_batch"
        logging.info(f"Testing {tool_name}...")
        result = fetch_slow_queries_batch(num_queries=NUM_QUERIES, time_range=TIME_RANGE)
        data = self._check_for_error(result, tool_name)
        # Returns dict with "queries" key
        self.assertIn("queries", data)
        self.assertIsInstance(data["queries"], list)
        if len(data["queries"]) > 0:
            self.assertIn("request_latency", data["queries"][0])

    def test_fetch_fastest_queries(self):
        tool_name = "fetch_fastest_queries"
        logging.info(f"Testing {tool_name}...")
        result = fetch_fastest_queries(num_queries=NUM_QUERIES, time_range=TIME_RANGE)
        data = self._check_for_error(result, tool_name)
        # Returns dict with "queries" key
        self.assertIn("queries", data)
        self.assertIsInstance(data["queries"], list)

    def test_check_kpi_compliance(self):
        tool_name = "check_kpi_compliance"
        logging.info(f"Testing {tool_name}...")
        result = check_kpi_compliance(time_range=TIME_RANGE)
        data = self._check_for_error(result, tool_name)
        self.assertTrue("status" in data or "compliance" in data)

    def test_cluster_slow_queries(self):
        tool_name = "cluster_slow_queries"
        logging.info(f"Testing {tool_name}...")
        result = cluster_slow_queries(num_queries=NUM_QUERIES, time_range=TIME_RANGE)
        # This one typically returns a string report
        if isinstance(result, str) and not result.startswith("Error") and not result.startswith("{"):
            pass 
        else:
            self._check_for_error(result, tool_name)

    def test_get_token_velocity(self):
        tool_name = "get_token_velocity"
        logging.info(f"Testing {tool_name}...")
        result = get_token_velocity(time_range=TIME_RANGE)
        data = self._check_for_error(result, tool_name)
        # Returns dict with metadata/statistics
        self.assertTrue("statistics" in data or "speed_breakdown" in data)

    def test_analyze_thinking_overhead(self):
        tool_name = "analyze_thinking_overhead"
        logging.info(f"Testing {tool_name}...")
        result = analyze_thinking_overhead(time_range=TIME_RANGE)
        data = self._check_for_error(result, tool_name)
        self.assertTrue("statistics" in data or "request_classification" in data)

    def test_detect_compute_inefficiency(self):
        tool_name = "detect_compute_inefficiency"
        logging.info(f"Testing {tool_name}...")
        result = detect_compute_inefficiency(time_range=TIME_RANGE)
        data = self._check_for_error(result, tool_name)
        self.assertIn("summary", data)

    def test_get_generation_config_comparison(self):
        tool_name = "get_generation_config_comparison"
        logging.info(f"Testing {tool_name}...")
        result = get_generation_config_comparison(time_range=TIME_RANGE)
        data = self._check_for_error(result, tool_name)
        self.assertTrue("configurations" in data)

    def test_analyze_config_correlation(self):
        tool_name = "analyze_config_correlation"
        logging.info(f"Testing {tool_name}...")
        result = analyze_config_correlation(time_range=TIME_RANGE)
        data = self._check_for_error(result, tool_name)
        self.assertIn("correlations", data)

    def test_get_config_outliers(self):
        tool_name = "get_config_outliers"
        logging.info(f"Testing {tool_name}...")
        result = get_config_outliers(time_range=TIME_RANGE)
        data = self._check_for_error(result, tool_name)
        self.assertIn("outliers", data)

    def test_get_agent_model_matrix(self):
        tool_name = "get_agent_model_matrix"
        logging.info(f"Testing {tool_name}...")
        result = get_agent_model_matrix(time_range=TIME_RANGE)
        data = self._check_for_error(result, tool_name)
        self.assertTrue(isinstance(data, list) or "matrix" in data)

if __name__ == "__main__":
    unittest.main(verbosity=2)

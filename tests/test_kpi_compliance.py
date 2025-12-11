
import unittest
import json
import logging
import sys
import os
from unittest.mock import patch, MagicMock

# Add the project root to python path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from agents.latency_analyzer.utils import check_kpi_compliance

class TestKPICompliance(unittest.TestCase):
    @patch('agents.latency_analyzer.utils.get_analysis_config')
    @patch('agents.latency_analyzer.utils.get_overall_statistics')
    @patch('agents.latency_analyzer.utils.get_agent_comparison')
    def test_check_kpi_compliance_per_agent(self, mock_get_agent_comparison, mock_get_overall_stats, mock_get_config):
        # Setup mocks
        mock_get_config.return_value = json.dumps({
            "kpis": {"mean_latency_target": 3.0, "p95_latency_target": 5.0},
            "agent_name": None
        })
        
        # Mock global stats
        mock_get_overall_stats.return_value = json.dumps({
            "latency": {"mean": 10.0, "p95": 20.0}
        })
        
        # Mock per-agent stats
        mock_get_agent_comparison.return_value = json.dumps({
            "agents": [
                {"agent_name": "fast_agent", "avg_latency": 1.0, "p95_latency": 2.0},
                {"agent_name": "slow_agent", "avg_latency": 10.0, "p95_latency": 20.0}
            ]
        })
        
        # Call the function
        result_json = check_kpi_compliance(time_range="24h")
        result = json.loads(result_json)
        
        # Verify structure
        self.assertIn("per_agent_compliance", result)
        self.assertEqual(len(result["per_agent_compliance"]), 2)
        
        # Verify fast agent
        fast_agent = next(a for a in result["per_agent_compliance"] if a["agent_name"] == "fast_agent")
        self.assertEqual(fast_agent["status"], "PASS")
        self.assertEqual(fast_agent["mean_status"], "PASS")
        self.assertEqual(fast_agent["p95_status"], "PASS")
        
        # Verify slow agent
        slow_agent = next(a for a in result["per_agent_compliance"] if a["agent_name"] == "slow_agent")
        self.assertEqual(slow_agent["status"], "FAIL")
        self.assertEqual(slow_agent["mean_status"], "FAIL")
        self.assertEqual(slow_agent["p95_status"], "FAIL")
        
        print("Test passed: per_agent_compliance field exists and has correct values.")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    unittest.main()

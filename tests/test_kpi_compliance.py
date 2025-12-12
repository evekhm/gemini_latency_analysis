
import unittest
from unittest.mock import patch, MagicMock
import json
import sys
import os

# Add updated path to sys.path to find utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../agents/parallel_latency_analyzer')))

import utils

class TestKPICompliance(unittest.TestCase):

    @patch('utils.get_analysis_config')
    @patch('utils.get_overall_statistics')
    @patch('utils.get_agent_comparison')
    @patch('utils.parse_time_range')
    def test_per_agent_compliance(self, mock_parse_time, mock_agent_comparison, mock_overall_stats, mock_config):
        # Mock Config
        mock_config.return_value = json.dumps({
            "kpis": {
                "mean_latency_target": 2.0,
                "p95_latency_target": 4.0,
                "per_agent": {
                    "slow_agent": {
                        "mean_latency_target": 5.0, # looser target
                        "p95_latency_target": 10.0
                    }
                }
            }
        })
        
        # Mock Overall Stats
        mock_overall_stats.return_value = json.dumps({
            "latency": {
                "mean": 1.5,
                "p95": 3.5
            },
            "metadata": {"time_range": "24h"}
        })

        # Mock Agent Comparison
        mock_agent_comparison.return_value = json.dumps({
            "agents": [
                {
                    "agent_name": "fast_agent",
                    "avg_latency": 1.0,
                    "p95_latency": 3.0
                },
                {
                    "agent_name": "slow_agent",
                    "avg_latency": 4.5, # Should pass custom target (5.0) but fail global (2.0)
                    "p95_latency": 9.0  # Should pass custom target (10.0) but fail global (4.0)
                },
                {
                    "agent_name": "failing_agent",
                    "avg_latency": 3.0, # Fails global (2.0)
                    "p95_latency": 5.0  # Fails global (4.0)
                }
            ]
        })
        
        # Run function
        result_json = utils.check_kpi_compliance()
        result = json.loads(result_json)
        
        # Verify Global Compliance
        self.assertEqual(result['compliance']['overall_status'], 'PASS') # 1.5 < 2.0 and 3.5 < 4.0
        
        # Verify Per-Agent Compliance
        per_agent = result.get('per_agent_compliance', [])
        self.assertEqual(len(per_agent), 3)
        
        # Fast Agent
        fast = next(a for a in per_agent if a['agent_name'] == 'fast_agent')
        self.assertEqual(fast['overall_status'], 'pass')
        self.assertEqual(fast['mean_latency']['target'], 2.0) # Global default
        
        # Slow Agent (Custom Settings)
        slow = next(a for a in per_agent if a['agent_name'] == 'slow_agent')
        self.assertEqual(slow['overall_status'], 'pass')
        self.assertEqual(slow['mean_latency']['target'], 5.0) # Custom override
        self.assertEqual(slow['mean_latency']['status'], 'pass') # 4.5 < 5.0
        
        # Failing Agent
        fail = next(a for a in per_agent if a['agent_name'] == 'failing_agent')
        self.assertEqual(fail['overall_status'], 'fail')
        self.assertEqual(fail['mean_latency']['target'], 2.0)
        self.assertEqual(fail['mean_latency']['status'], 'fail') # 3.0 > 2.0

if __name__ == '__main__':
    unittest.main()

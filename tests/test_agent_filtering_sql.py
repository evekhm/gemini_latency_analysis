
import unittest
import json
import os
from unittest.mock import patch, MagicMock
import sys

# Add parent directory to path to import utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.parallel_latency_analyzer import utils

class TestAgentFilteringSQL(unittest.TestCase):
    def setUp(self):
        # Reset any global state if necessary
        pass

    @patch('agents.parallel_latency_analyzer.utils._load_config_data')
    @patch('agents.parallel_latency_analyzer.utils.execute_bigquery')
    def test_global_filter_included(self, mock_execute, mock_config):
        """Test that agents_included adds correct IN clause"""
        # Mock config
        mock_config.return_value = {
            "filters": {
                "agents_included": "agent_a, agent_b",
                "agents_excluded": ""
            }
        }
        
        # Call a tool that hits BigQuery
        utils.get_overall_statistics(time_range="24h")
        
        # Verify the query passed to execute_bigquery
        args, _ = mock_execute.call_args
        query = args[0]
        
        # Should include filter
        self.assertIn("T.agent_name IN ('agent_a', 'agent_b')", query)
        # Should NOT include excluded filter
        self.assertNotIn("NOT IN", query)

    @patch('agents.parallel_latency_analyzer.utils._load_config_data')
    @patch('agents.parallel_latency_analyzer.utils.execute_bigquery')
    def test_global_filter_excluded(self, mock_execute, mock_config):
        """Test that agents_excluded adds correct NOT IN clause"""
        mock_config.return_value = {
            "filters": {
                "agents_included": "",
                "agents_excluded": "agent_x"
            }
        }
        
        utils.get_overall_statistics(time_range="24h")
        
        args, _ = mock_execute.call_args
        query = args[0]
        
        self.assertIn("T.agent_name NOT IN ('agent_x')", query)

    @patch('agents.parallel_latency_analyzer.utils._load_config_data')
    @patch('agents.parallel_latency_analyzer.utils.execute_bigquery')
    def test_global_filter_both_preference(self, mock_execute, mock_config):
        """Test that included takes precedence over excluded"""
        mock_config.return_value = {
            "filters": {
                "agents_included": "agent_a",
                "agents_excluded": "agent_b"
            }
        }
        
        utils.get_overall_statistics(time_range="24h")
        
        args, _ = mock_execute.call_args
        query = args[0]
        
        self.assertIn("IN ('agent_a')", query)
        self.assertNotIn("NOT IN", query)

    @patch('agents.parallel_latency_analyzer.utils._load_config_data')
    @patch('agents.parallel_latency_analyzer.utils.execute_bigquery')
    def test_no_filter(self, mock_execute, mock_config):
        """Test that empty filters produce no SQL change"""
        mock_config.return_value = {
            "filters": {
                "agents_included": "",
                "agents_excluded": ""
            }
        }
        
        utils.get_overall_statistics(time_range="24h")
        
        args, _ = mock_execute.call_args
        query = args[0]
        
        # Should not have agent filtering clauses (unless specific agent arg was passed, which defaults to None)
        self.assertNotIn("T.agent_name IN", query)
        self.assertNotIn("T.agent_name NOT IN", query)

    @patch('agents.parallel_latency_analyzer.utils._load_config_data')
    @patch('agents.parallel_latency_analyzer.utils.execute_bigquery')
    def test_filter_propagation_to_distribution(self, mock_execute, mock_config):
        """Verify filter is applied to get_latency_distribution too"""
        mock_config.return_value = {
            "filters": {
                "agents_included": "test_agent",
            }
        }
        
        utils.get_latency_distribution(time_range="24h")
        
        args, _ = mock_execute.call_args
        query = args[0]
        
        self.assertIn("IN ('test_agent')", query)

    @patch('agents.parallel_latency_analyzer.utils._load_config_data')
    @patch('agents.parallel_latency_analyzer.utils.execute_bigquery')
    def test_filter_concurrent_impact(self, mock_execute, mock_config):
        """Verify filter is applied to get_concurrent_request_impact which uses model_name pattern"""
        mock_config.return_value = {
            "filters": {
                "agents_included": "test_agent_concurrent",
            }
        }
        
        # Mock dataframe return to avoid errors in processing
        mock_execute.return_value = MagicMock()
        mock_execute.return_value.empty = True
        
        utils.get_concurrent_request_impact(time_range="24h")
        
        args, _ = mock_execute.call_args
        query = args[0]
        
        self.assertIn("IN ('test_agent_concurrent')", query)

if __name__ == '__main__':
    unittest.main()

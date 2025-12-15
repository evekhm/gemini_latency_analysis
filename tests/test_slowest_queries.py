
import unittest
import json
import logging
import sys
import os
from unittest.mock import patch, MagicMock, ANY

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from agents.parallel_latency_analyzer.utils import get_slowest_queries

class TestSlowestQueries(unittest.TestCase):
    def setUp(self):
        # Setup common mocks
        self.mock_time_range_patcher = patch('agents.parallel_latency_analyzer.utils.parse_time_range')
        self.mock_parse_time = self.mock_time_range_patcher.start()
        self.mock_parse_time.return_value = json.dumps({
            "start_date": "2023-01-01 00:00:00",
            "end_date": "2023-01-02 00:00:00"
        })

        self.mock_bq_patcher = patch('agents.parallel_latency_analyzer.utils.execute_bigquery')
        self.mock_execute_bq = self.mock_bq_patcher.start()

        # Mock environment variables used in utils
        # Note: PROJECT_ID and DATASET_ID are globals in utils, so we must patch them there,
        # not just os.environ, because they are loaded at import time.
        self.project_id_patcher = patch('agents.parallel_latency_analyzer.utils.PROJECT_ID', 'test_project')
        self.dataset_id_patcher = patch('agents.parallel_latency_analyzer.utils.DATASET_ID', 'test_dataset')
        self.view_id_patcher = patch('agents.parallel_latency_analyzer.utils.VIEW_ID', 'llm_logging_view')
        
        self.project_id_patcher.start()
        self.dataset_id_patcher.start()
        self.view_id_patcher.start()
        
        # We don't need to mock get_table_list or build_multi_table_source anymore
        # as the new implementation uses the View directly.

    def tearDown(self):
        self.mock_time_range_patcher.stop()
        self.mock_bq_patcher.stop()
        self.project_id_patcher.stop()
        self.dataset_id_patcher.stop()
        self.view_id_patcher.stop()

    def test_get_slowest_queries_call_signature(self):
        """Test that get_slowest_queries calls execute_bigquery directly"""
        # Mock empty dataframe response
        mock_df = MagicMock()
        mock_df.empty = True
        self.mock_execute_bq.return_value = mock_df

        get_slowest_queries(num_queries=10)

        # Check that execute_bigquery was called
        self.mock_execute_bq.assert_called_once()
        
        # Verify the query contains the View reference
        args, _ = self.mock_execute_bq.call_args
        query = args[0]
        self.assertIn("FROM \n            `test_project.test_dataset.llm_logging_view` AS T", query)

    def test_get_slowest_queries_agent_filter(self):
        """Test that agent_name filter is applied correctly via SQL"""
        mock_df = MagicMock()
        mock_df.empty = True
        self.mock_execute_bq.return_value = mock_df

        get_slowest_queries(agent_name="test_agent")
        
        args, _ = self.mock_execute_bq.call_args
        query = args[0]
        self.assertIn("T.agent_name = 'test_agent'", query)

    def test_get_slowest_queries_sql_structure(self):
        """Test that the final SQL query structure is valid"""
        mock_df = MagicMock()
        mock_df.empty = True
        self.mock_execute_bq.return_value = mock_df

        get_slowest_queries(num_queries=50)
        
        args, _ = self.mock_execute_bq.call_args
        query = args[0]
        
        # Check for key tokens
        self.assertIn("SELECT", query)
        self.assertIn("request_id", query)
        self.assertIn("request_latency AS latency", query)
        self.assertIn("ORDER BY latency DESC", query)
        self.assertIn("LIMIT 50", query)

if __name__ == '__main__':
    logging.basicConfig(level=logging.ERROR)
    unittest.main()

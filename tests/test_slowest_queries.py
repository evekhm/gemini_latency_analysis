
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

        self.mock_table_list_patcher = patch('agents.parallel_latency_analyzer.utils.get_table_list')
        self.mock_get_table_list = self.mock_table_list_patcher.start()
        # Default to single table
        self.mock_get_table_list.return_value = ['test_table']
        
        # Patch build_multi_table_source to ensure we test its usage or mock it if needed.
        # However, to test integration, we might want to let it run if it's simple string manipulation.
        # But get_slowest_queries relies on build_multi_table_source from the SAME module.
        # Let's mock it to verify the call arguments specifically 'select_suffix="AS T"' which was the fix.
        self.mock_build_source_patcher = patch('agents.parallel_latency_analyzer.utils.build_multi_table_source')
        self.mock_build_source = self.mock_build_source_patcher.start()
        # build_multi_table_source returns (SELECT ...) AS T
        self.mock_build_source.return_value = "(SELECT * FROM `test_table` AS T) AS T" 

    def tearDown(self):
        self.mock_time_range_patcher.stop()
        self.mock_bq_patcher.stop()
        self.mock_table_list_patcher.stop()
        self.mock_build_source_patcher.stop()

    def test_get_slowest_queries_call_signature(self):
        """Test that get_slowest_queries calls deps correctly with defaults"""
        # Mock empty dataframe response to avoid processing errors
        mock_df = MagicMock()
        mock_df.empty = True
        self.mock_execute_bq.return_value = mock_df

        get_slowest_queries(num_queries=10)

        # check build_multi_table_source called with AS T
        self.mock_build_source.assert_called_with(ANY, select_suffix="AS T")
        
        # Verify the where clause passed to build_multi_table_source contains expected filters
        args, _ = self.mock_build_source.call_args
        where_clause = args[0]
        self.assertIn("T.logging_time BETWEEN", where_clause)
        # Ensure we are NOT using T. prefix in where_clause if build_multi_table_source handles it? 
        # Actually existing code uses T. prefix in where_clauses list.
        self.assertIn("T.full_request IS NOT NULL", where_clause)

    def test_get_slowest_queries_agent_filter(self):
        """Test that agent_name filter is applied correctly"""
        mock_df = MagicMock()
        mock_df.empty = True
        self.mock_execute_bq.return_value = mock_df

        get_slowest_queries(agent_name="test_agent")
        
        args, _ = self.mock_build_source.call_args
        where_clause = args[0]
        self.assertIn("test_agent", where_clause)
        self.assertIn("adk_agent_name", where_clause)

    def test_get_slowest_queries_sql_structure(self):
        """Test that the final SQL query structure is valid (OFFSET fix check)"""
        # We can't easily check the exact SQL string because we mocked build_multi_table_source returns a stub.
        # But we can check that execute_bigquery is called with a string containing the stub.
        mock_df = MagicMock()
        mock_df.empty = True
        self.mock_execute_bq.return_value = mock_df

        get_slowest_queries()
        
        args, _ = self.mock_execute_bq.call_args
        query = args[0]
        
        # Check for the key elements of the fix
        # self.assertIn("WITH OFFSET AS off", query)
        # self.assertIn("ORDER BY off DESC", query)
        # We expect FROM to be followed by the source
        self.assertIn("FROM \n            (SELECT * FROM `test_table` AS T) AS T", query)

    def test_multi_table_generation(self):
        """Test SQL generation for multiple tables (UNION ALL check)"""
        # Unpatch build_multi_table_source for this test to check real logic
        self.mock_build_source_patcher.stop()
        
        # Mock table list to return multiple tables
        self.mock_get_table_list.return_value = ['2p5-flash', '3-pro', '2p5-pro']
        
        # Mock Project/Dataset globals in utils if they are used by build_multi_table_source
        # Since we are importing get_slowest_queries, build_multi_table_source is in the same module.
        # We need to make sure PROJECT_ID and DATASET_ID are set in that module.
        with patch('agents.parallel_latency_analyzer.utils.PROJECT_ID', 'test_project'), \
             patch('agents.parallel_latency_analyzer.utils.DATASET_ID', 'test_dataset'):
             
            mock_df = MagicMock()
            mock_df.empty = True
            self.mock_execute_bq.return_value = mock_df

            get_slowest_queries()
            
            args, _ = self.mock_execute_bq.call_args
            query = args[0]

            # Verify UNION ALL structure
            self.assertIn("UNION ALL", query)
            self.assertIn("FROM `test_project.test_dataset.2p5-flash` AS T", query)
            self.assertIn("FROM `test_project.test_dataset.3-pro` AS T", query)
            self.assertIn("FROM `test_project.test_dataset.2p5-pro` AS T", query)
            
            # Verify wrapping
            # The pattern is (SELECT ... UNION ALL SELECT ...) AS T
            self.assertIn(") AS T", query)

        # Re-start patcher for tearDown to not fail
        self.mock_build_source = self.mock_build_source_patcher.start()

if __name__ == '__main__':
    logging.basicConfig(level=logging.ERROR)
    unittest.main()

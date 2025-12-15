
import unittest
from unittest.mock import patch, MagicMock
import json
import sys
import os
import pandas as pd

# Add updated path to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.parallel_latency_analyzer import utils

class TestModelCleanIntegration(unittest.TestCase):

    @patch('agents.parallel_latency_analyzer.utils.execute_bigquery')
    @patch('agents.parallel_latency_analyzer.utils.get_table_list')
    @patch('agents.parallel_latency_analyzer.utils.parse_time_range')
    def test_fetch_fastest_queries_cleaning(self, mock_parse, mock_tables, mock_bq):
        mock_parse.return_value = json.dumps({"start_date": "2023-01-01", "end_date": "2023-01-02"})
        mock_tables.return_value = ["table1"]
        
        # Mock DataFrame with full model path
        mock_df = pd.DataFrame([{
            'logging_time': pd.Timestamp('2023-01-01 12:00:00'),
            'request_id': 'req1',
            'full_request': '{}',
            'full_response': '{}',
            'model': 'publishers/google/models/gemini-2.5-pro',
            'agent_name': 'agent1',
            'request_latency': 0.5,
            'thoughts_token_count': 10,
            'output_token_count': 100,
            'prompt_token_count': 50,
            'total_token_count': 160,
            'query_preview': 'test'
        }])
        mock_bq.return_value = mock_df
        
        result_json = utils.fetch_fastest_queries()
        result = json.loads(result_json)
        
        self.assertEqual(result['queries'][0]['model'], 'gemini-2.5-pro')

    @patch('agents.parallel_latency_analyzer.utils.execute_bigquery')
    @patch('agents.parallel_latency_analyzer.utils.get_table_list')
    @patch('agents.parallel_latency_analyzer.utils.parse_time_range')
    def test_fetch_slow_queries_batch_cleaning(self, mock_parse, mock_tables, mock_bq):
        mock_parse.return_value = json.dumps({"start_date": "2023-01-01", "end_date": "2023-01-02"})
        mock_tables.return_value = ["table1"]
        
        mock_df = pd.DataFrame([{
            'logging_time': pd.Timestamp('2023-01-01 12:00:00'),
            'request_id': 'req2',
            'full_request': '{}',
            'full_response': '{}',
            'model': 'publishers/google/models/gemini-1.5-flash',
            'agent_name': 'agent2',
            'request_latency': 10.5,
            'thoughts_token_count': 100,
            'output_token_count': 1000,
            'prompt_token_count': 500,
            'total_token_count': 1600,
            'request_preview': 'slow test'
        }])
        mock_bq.return_value = mock_df
        
        result_json = utils.fetch_slow_queries_batch()
        result = json.loads(result_json)
        
        self.assertEqual(result['queries'][0]['model'], 'gemini-1.5-flash')

    @patch('google.cloud.bigquery.Client')
    @patch('agents.parallel_latency_analyzer.utils.get_table_list')
    def test_fetch_single_query_cleaning(self, mock_tables, mock_client_cls):
        mock_tables.return_value = ["table1"]
        
        # Mock BigQuery Client and Query Job
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        
        mock_df = pd.DataFrame([{
            'logging_time': pd.Timestamp('2023-01-01 12:00:00'),
            'request_id': 'req3',
            'full_request': '{}',
            'full_response': '{}',
            'model': 'models/gemini-pro',
            'adk_agent_name': 'agent3',
            'request_latency': 1.5,
            'thoughts_token_count': 0,
            'output_token_count': 50,
            'prompt_token_count': 10,
            'total_token_count': 60
        }])
        
        mock_client.query.return_value.to_dataframe.return_value = mock_df
        
        result_json = utils.fetch_single_query("req3")
        result = json.loads(result_json)
        
        self.assertEqual(result['model'], 'gemini-pro')

if __name__ == '__main__':
    unittest.main()

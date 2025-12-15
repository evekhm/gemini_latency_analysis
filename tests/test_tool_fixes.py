
import unittest
import json
import pandas as pd
import sys
import os
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from agents.parallel_latency_analyzer.utils import (
    get_hourly_patterns,
    get_model_comparison,
    get_agent_model_matrix,
    get_slowest_queries
)

class TestToolReliability(unittest.TestCase):
    def setUp(self):
        self.mock_bq_patcher = patch('agents.parallel_latency_analyzer.utils.execute_bigquery')
        self.mock_execute_bq = self.mock_bq_patcher.start()
        
        self.mock_time_range_patcher = patch('agents.parallel_latency_analyzer.utils.parse_time_range')
        self.mock_parse_time = self.mock_time_range_patcher.start()
        self.mock_parse_time.return_value = json.dumps({
            "start_date": "2023-01-01 00:00:00",
            "end_date": "2023-01-02 00:00:00"
        })
        
        self.mock_get_table_list = patch('agents.parallel_latency_analyzer.utils.get_table_list')
        self.mock_table_list = self.mock_get_table_list.start()
        self.mock_table_list.return_value = ['test_table']

    def tearDown(self):
        self.mock_bq_patcher.stop()
        self.mock_time_range_patcher.stop()
        self.mock_get_table_list.stop()

    def test_hourly_patterns_p95(self):
        # Mock DF return with p95 column
        mock_df = pd.DataFrame([{
            'hour': 10, 'day_of_week': 2, 'day_type': 'working', 
            'request_count': 100, 'avg_latency': 1.5, 'p95_latency': 2.5
        }])
        self.mock_execute_bq.return_value = mock_df
        
        result_json = get_hourly_patterns()
        result = json.loads(result_json)
        
        # Check if p95_latency is present in output
        self.assertTrue('p95_latency' in result['working_days'][0])
        self.assertEqual(result['working_days'][0]['p95_latency'], 2.5)

    def test_model_comparison_exists(self):
        # Simply verifying the tool runs and generates SQL with model extraction
        mock_df = pd.DataFrame([{
            'publisher': 'google', 'model_name': 'gemini-pro', 'full_model_path': 'publishers/google/models/gemini-pro',
            'total_calls': 50, 'avg_latency': 1.0, 'p95_latency': 2.0,
            'avg_prompt_token_count': 100, 'p95_prompt_token_count': 150,
            'avg_candidates_token_count': 50, 'p95_candidates_token_count': 80,
            'avg_thoughts_token_count': 0, 'p95_thoughts_token_count': 0,
            'avg_total_token_count': 150, 'total_token_count_sum': 7500, 'avg_tpot': 0.05
        }])
        self.mock_execute_bq.return_value = mock_df
        
        result_json = get_model_comparison()
        result = json.loads(result_json)
        
        self.assertEqual(result['models'][0]['model_name'], 'gemini-pro')
        
    def test_agent_model_matrix_exists(self):
        mock_df = pd.DataFrame([{
            'agent_name': 'writer', 'publisher': 'google', 'model_name': 'gemini-pro', 'full_model_path': 'publishers/google/models/gemini-pro',
            'total_calls': 50, 'avg_latency': 1.0, 'p95_latency': 2.0,
            'avg_prompt_token_count': 100, 'p95_prompt_token_count': 150,
            'avg_candidates_token_count': 50, 'p95_candidates_token_count': 80,
            'avg_thoughts_token_count': 0, 'p95_thoughts_token_count': 0, 
            'avg_tpot': 0.05
        }])
        self.mock_execute_bq.return_value = mock_df
        
        result_json = get_agent_model_matrix()
        result = json.loads(result_json)
        
        self.assertIn('writer', result['matrix'])
        self.assertIn('gemini-pro', result['matrix']['writer'])

    def test_slowest_queries_robustness(self):
        # Ensure it handles extraction logic
        mock_df = pd.DataFrame([{
            'request_id': '123', 'logging_time': pd.Timestamp('2023-01-01'), 
            'latency': 5.0, 'model': 'gemini-pro', 'agent_name': 'writer',
            'prompt_token_count': 10, 'candidates_token_count': 10, 'thoughts_token_count': 0, 'total_token_count': 20,
            'last_user_message': 'Why so slow?'
        }])
        self.mock_execute_bq.return_value = mock_df
        
        result_json = get_slowest_queries()
        result = json.loads(result_json)
        self.assertEqual(len(result['slowest_queries']), 1)

if __name__ == '__main__':
    unittest.main()

# tests/test_slow_query_analyzer.py
import unittest
from unittest.mock import MagicMock, patch
import json
import sys
import os
import sys

# Set env var for testing before imports
os.environ['MODEL'] = 'gemini-pro'

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.slow_query_analyzer.utils import fetch_slow_queries

class TestSlowQueryAnalyzer(unittest.TestCase):

    @patch('agents.slow_query_analyzer.utils.bigquery.Client')
    @patch('agents.slow_query_analyzer.utils.PROJECT_ID', 'test-project')
    def test_fetch_slow_queries(self, mock_client_cls):
        # Setup mock
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        
        mock_query_job = MagicMock()
        mock_client.query.return_value = mock_query_job
        
        # Mock results
        mock_row = {
            'logging_time': '2023-10-27 10:00:00',
            'request_id': 'req-123',
            'full_request': '{"text": "hello"}',
            'full_response': '{"text": "world"}',
            'model': 'gemini-pro',
            'adk_agent_name': 'test-agent',
            'request_latency_seconds': 10.5,
            'thoughts_token_count': 100,
            'output_token_count': 50,
            'prompt_token_count': 20,
            'total_token_count': 170
        }
        mock_query_job.result.return_value = [mock_row]

        # Execute
        result_json = fetch_slow_queries(limit=5)
        
        # Verify
        self.assertTrue(mock_client.query.called)
        call_args = mock_client.query.call_args[0][0]
        self.assertIn("LIMIT 5", call_args)
        self.assertIn("ORDER BY\n      request_latency_seconds DESC", call_args)
        
        result = json.loads(result_json)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['request_id'], 'req-123')
        print("Test passed: fetch_slow_queries returned expected JSON.")

if __name__ == '__main__':
    unittest.main()

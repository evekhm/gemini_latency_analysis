#!/usr/bin/env python3
"""
Test multi-table support functionality.
"""

import os
import sys

from dotenv import load_dotenv
__dir__ = os.path.dirname(__file__)
load_dotenv(dotenv_path=os.path.join(__dir__, "../.env"))

def test_single_table():
    """Test parsing a single table name."""
    # Import here to avoid .env file loading interference
    os.environ['AGENT_TABLE_ID'] = '2p5-pro'
    
    # Reimport to pick up the new env var
    import importlib
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from agents.parallel_latency_analyzer import utils
    importlib.reload(utils)
    
    tables = utils.get_table_list()
    assert tables == ['2p5-pro'], f"Expected ['2p5-pro'], got {tables}"
    print("✓ Single table test passed")

def test_multiple_tables():
    """Test parsing comma-separated table names."""
    os.environ['AGENT_TABLE_ID'] = '2p5-pro, 2p5-flash'
    
    import importlib
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from agents.parallel_latency_analyzer import utils
    importlib.reload(utils)
    
    tables = utils.get_table_list()
    expected = ['2p5-pro', '2p5-flash']
    assert tables == expected, f"Expected {expected}, got {tables}"
    print("✓ Multiple tables test passed")

def test_multiple_tables_extra_whitespace():
    """Test parsing with extra whitespace."""
    os.environ['AGENT_TABLE_ID'] = '  2p5-pro  ,  2p5-flash  ,  1p5-pro  '
    
    import importlib
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from agents.parallel_latency_analyzer import utils
    importlib.reload(utils)
    
    tables = utils.get_table_list()
    expected = ['2p5-pro', '2p5-flash', '1p5-pro']
    assert tables == expected, f"Expected {expected}, got {tables}"
    print("✓ Whitespace handling test passed")


if __name__ == '__main__':
    print("Running multi-table support tests...")
    test_single_table()
    test_multiple_tables()
    test_multiple_tables_extra_whitespace()

    print("\\n✅ All tests passed!")


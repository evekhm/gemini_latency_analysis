#!/usr/bin/env python3
"""
Test script to verify model extraction and per-model analysis tools.

This script tests:
1. Model extraction from BigQuery model field
2. get_model_comparison() tool
3. get_agent_model_matrix() tool
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.parallel_latency_analyzer.utils import (
    get_model_comparison,
    get_agent_model_matrix,
    execute_bigquery
)
import json
from dotenv import load_dotenv
__dir__ = os.path.dirname(__file__)
load_dotenv(dotenv_path=os.path.join(__dir__, "../.env"))

PROJECT_ID = os.getenv('PROJECT_ID')
DATASET_ID = os.getenv('DATASET_ID')
TABLE_ID = os.getenv('AGENT_TABLE_ID')



def test_model_extraction():
    """Test SQL model extraction from sample data."""
    print("\n" + "="*60)
    print("TEST 1: Model Extraction SQL")
    print("="*60)
    

    
    if not TABLE_ID:
        print("❌ AGENT_TABLE_ID not set")
        return False
    
    # Test query to verify extraction works
    # Handle multi-table configuration by picking the first table
    from agents.parallel_latency_analyzer.utils import get_table_list
    header_table = get_table_list()[0]
    
    query = f"""
    SELECT 
      T.model AS full_path,
      SPLIT(T.model, '/')[SAFE_OFFSET(1)] AS publisher,
      SPLIT(T.model, '/')[SAFE_OFFSET(3)] AS model_name
    FROM `{PROJECT_ID}.{DATASET_ID}.{header_table}` AS T
    WHERE T.model IS NOT NULL
    LIMIT 10
    """
    
    try:
        df = execute_bigquery(query, timeout=30)
        
        if df.empty:
            print("⚠️  No data found")
            return False
        
        print(f"\n✅ Successfully extracted {len(df)} model records:\n")
        for _, row in df.iterrows():
            print(f"  Full Path: {row['full_path']}")
            print(f"  Publisher: {row['publisher']}")
            print(f"  Model: {row['model_name']}")
            print()
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False


def test_model_comparison():
    """Test get_model_comparison() tool."""
    print("\n" + "="*60)
    print("TEST 2: get_model_comparison() Tool")
    print("="*60)
    
    try:
        result_json = get_model_comparison(time_range="7d")
        result = json.loads(result_json)
        
        if "error" in result:
            print(f"❌ Error: {result['error']}")
            return False
        
        print(f"\n✅ Successfully analyzed {result['metadata']['total_models']} models:\n")
        
        for model in result['models'][:5]:  # Show top 5
            print(f"  Model: {model['model_name']}")
            print(f"    Publisher: {model['publisher']}")
            print(f"    Calls: {model['total_calls']}")
            print(f"    Avg Latency: {model['avg_latency']:.2f}s")
            print(f"    P95 Latency: {model['p95_latency']:.2f}s" if model['p95_latency'] else "")
            print()
        
        print(f"Insights:")
        print(f"  Fastest: {result['insights']['fastest_model']}")
        print(f"  Slowest: {result['insights']['slowest_model']}")
        print(f"  Most Active: {result['insights']['most_active_model']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_agent_model_matrix():
    """Test get_agent_model_matrix() tool."""
    print("\n" + "="*60)
    print("TEST 3: get_agent_model_matrix() Tool")
    print("="*60)
    
    try:
        result_json = get_agent_model_matrix(time_range="7d")
        result = json.loads(result_json)
        
        if "error" in result:
            print(f"❌ Error: {result['error']}")
            return False
        
        print(f"\n✅ Successfully analyzed agent×model matrix:\n")
        print(f"  Agents: {result['metadata']['total_agents']}")
        print(f"  Models: {result['metadata']['total_models']}")
        print(f"  Combinations: {result['metadata']['total_combinations']}")
        print()
        
        print("Model Switching Detection:")
        if result['insights']['model_switching_detected']:
            print(f"  ⚠️  {len(result['insights']['agents_switching_models'])} agents switch models:")
            for agent in result['insights']['agents_switching_models'][:3]:
                models = result['agents_with_multiple_models'][agent]
                print(f"    - {agent}: uses {models}")
        else:
            print("  ✅ No model switching detected")
        
        print()
        print("Performance Extremes:")
        if result['insights']['slowest_combination']:
            slow = result['insights']['slowest_combination']
            print(f"  Slowest: {slow['agent']} + {slow['model']} = {slow['latency']:.2f}s")
        if result['insights']['fastest_combination']:
            fast = result['insights']['fastest_combination']
            print(f"  Fastest: {fast['agent']} + {fast['model']} = {fast['latency']:.2f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "="*60)
    print("TESTING PER-MODEL ANALYSIS TOOLS")
    print("="*60)
    
    results = []
    
    # Run tests
    results.append(("Model Extraction SQL", test_model_extraction()))
    results.append(("get_model_comparison()", test_model_comparison()))
    results.append(("get_agent_model_matrix()", test_agent_model_matrix()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n🎉 All tests passed! Per-model analysis is ready.")
        sys.exit(0)
    else:
        print("\n⚠️  Some tests failed. Please review above.")
        sys.exit(1)

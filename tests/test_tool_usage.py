import sys
import os
import json
import logging

# Ensure project root is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.parallel_latency_analyzer import utils

def test_tool_tracking():
    print("Testing tool tracking...")
    
    # 1. Call a decorated tool
    try:
        # get_analysis_metadata is decorated and simple (no args needed)
        # It relies on env vars which should be loaded by utils
        print("Calling get_analysis_metadata...")
        try:
            utils.get_analysis_metadata()
        except Exception as e:
            # Even if it fails (due to missing creds), it should track usage
            print(f"Tool call failed (expected if no creds): {e}")
            pass
            
        # 2. Call get_tool_usage_report
        print("Calling get_tool_usage_report...")
        report_json = utils.get_tool_usage_report()
        report = json.loads(report_json)
        
        print(f"Report received: {json.dumps(report, indent=2)}")
        
        # Verify
        found = False
        for entry in report:
            if entry['tool'] == 'get_analysis_metadata':
                found = True
                print(f"Found stats for get_analysis_metadata: {entry}")
                break
        
        if found:
            print("SUCCESS: Tool usage tracked.")
        else:
            print("FAILURE: Tool usage NOT tracked.")
            # Print keys in stats
            from agents.parallel_latency_analyzer.telemetry import get_tool_stats
            print(f"Actual stats keys: {get_tool_stats().keys()}")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    test_tool_tracking()

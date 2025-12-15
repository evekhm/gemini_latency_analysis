#!/usr/bin/env python3
"""
Test script to verify configuration handling.
Checks that all config fields are properly read and accessible.
"""

import json
import os


def test_config_handling():
    """Test that configuration is properly read from the config file."""
    
    print("=" * 80)
    print("CONFIGURATION HANDLING TEST")
    print("=" * 80)
    print()
    
    # Find the config file
    config_path = os.path.join(os.path.dirname(__file__), '../autonomous_analysis_90d.json')
    
    if not os.path.exists(config_path):
        print(f"✗ ERROR: Config file not found at {config_path}")
        return False
    
    print(f"1. Reading config from: {config_path}")
    
    # Read the config file
    try:
        with open(config_path, 'r') as f:
            data = json.load(f)
            config = data.get("config", data)
        print("   ✓ Config file successfully read and parsed\n")
    except Exception as e:
        print(f"   ✗ ERROR: Failed to read config: {e}\n")
        return False
    
    # Check for expected fields
    print("2. Checking for expected configuration fields:")
    expected_fields = {
        "time_period_days": "str",
        "analysis_scope": "str",
        "kpis": "dict",
        "num_slowest_queries": "int",
    }
    
    all_present = True
    for field, expected_type in expected_fields.items():
        if field in config:
            value = config[field]
            actual_type = type(value).__name__ if value is not None else "null"
            print(f"   ✓ '{field}': {value} (type: {actual_type})")
        else:
            print(f"   ✗ MISSING: '{field}' not found in config")
            all_present = False
    
    print()
    
    # Check nested KPI fields
    if "kpis" in config and isinstance(config["kpis"], dict):
        print("3. Checking KPI sub-fields:")
        kpi_fields = ["mean_latency_target", "p95_latency_target"]
        for field in kpi_fields:
            if field in config["kpis"]:
                value = config["kpis"][field]
                print(f"   ✓ kpis.{field}: {value}")
            else:
                print(f"   ✗ MISSING: kpis.{field}")
                all_present = False
    else:
        print("3. ✗ ERROR: 'kpis' is not a dict or is missing")
        all_present = False
    
    print()
    
    # Simulate what agent would do
    print("4. Simulating agent usage:")
    print(f"   config_json = get_analysis_config()")
    print(f"   config = json.loads(config_json)")
    print(f"   time_range = config['time_period_days']  # → '{config.get('time_period_days')}'")
    print(f"   kpi_target = config['kpis']['mean_latency_target']  # → {config.get('kpis', {}).get('mean_latency_target')}")
    print(f"   scope = config.get('analysis_scope', 'standard')  # → '{config.get('analysis_scope', 'standard')}'")
    print()
    
    # Print final summary
    print("=" * 80)
    if all_present:
        print("✅ ALL CONFIGURATION FIELDS PRESENT AND ACCESSIBLE")
    else:
        print("❌ SOME CONFIGURATION FIELDS ARE MISSING")
    print("=" * 80)
    print()
    
    # Print formatted config for review
    print("Full Configuration:")
    print(json.dumps(config, indent=2))
    
    return all_present

if __name__ == "__main__":
    import sys
    success = test_config_handling()
    sys.exit(0 if success else 1)


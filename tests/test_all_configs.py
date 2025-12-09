#!/usr/bin/env python3
"""
Test all configuration files to ensure they have the expected structure.
"""

import json
import os
import glob

def test_single_config(filepath):
    """Test a single config file."""
    print(f"\nTesting: {filepath}")
    print("-" * 80)
    
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
            config = data.get("config")
            
        if config is None:
            print("  ⚠️  No 'config' section found (may be using query-only format)")
            return True
            
        # Check for expected fields
        required_fields = ["analysis_scope"]
        optional_fields = ["time_period_days", "kpis", "num_slowest_queries", "agent_name"]
        
        all_good = True
        
        # Check required fields
        for field in required_fields:
            if field in config:
                print(f"  ✓ {field}: {config[field]}")
            else:
                print(f"  ✗ MISSING: {field}")
                all_good = False
        
        # Check optional fields
        for field in optional_fields:
            if field in config:
                value = config[field]
                if field == "kpis" and isinstance(value, dict):
                    print(f"  ✓ {field}:")
                    for k, v in value.items():
                        print(f"      - {k}: {v}")
                else:
                    print(f"  ✓ {field}: {value}")
        
        return all_good
        
    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        return False

def test_all_configs():
    """Test all JSON config files in the directory."""
    print("=" * 80)
    print("TESTING ALL CONFIGURATION FILES")
    print("=" * 80)
    
    # Find all JSON files
    json_files = glob.glob("*.json")
    config_files = [f for f in json_files if "analysis" in f or "kpi" in f or "cost" in f]
    
    if not config_files:
        print("\nNo configuration files found!")
        return False
    
    print(f"\nFound {len(config_files)} configuration files:")
    for f in config_files:
        print(f"  - {f}")
    
    all_passed = True
    for config_file in config_files:
        if not test_single_config(config_file):
            all_passed = False
    
    print("\n" + "=" * 80)
    if all_passed:
        print("✅ ALL CONFIGURATION FILES VALID")
    else:
        print("❌ SOME CONFIGURATION FILES HAVE ISSUES")
    print("=" * 80)
    
    return all_passed

if __name__ == "__main__":
    import sys
    success = test_all_configs()
    sys.exit(0 if success else 1)

#!/usr/bin/env python3
"""
Run all tests in the tests/ directory.
"""

import os
import sys
import subprocess
from pathlib import Path

def run_test(test_file):
    """Run a single test file and return results."""
    print(f"\n{'='*80}")
    print(f"Running: {test_file.name}")
    print('='*80)
    
    try:
        result = subprocess.run(
            [sys.executable, str(test_file)],
            cwd=test_file.parent.parent,  # Run from project root
            capture_output=True,
            text=True,
            timeout=30
        )
        
        # Print output
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr, file=sys.stderr)
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print(f"✗ TIMEOUT: Test took longer than 30 seconds")
        return False
    except Exception as e:
        print(f"✗ ERROR: {e}")
        return False

def run_all_tests():
    """Discover and run all test files."""
    # Get the tests directory
    tests_dir = Path(__file__).parent
    
    print("="*80)
    print("TEST RUNNER - Configuration Tests")
    print("="*80)
    print(f"\nTest directory: {tests_dir}")
    
    # Find all test files
    test_files = sorted(tests_dir.glob("test_*.py"))
    
    if not test_files:
        print("\n✗ No test files found!")
        return False
    
    print(f"Found {len(test_files)} test file(s):")
    for f in test_files:
        print(f"  - {f.name}")
    
    # Run each test
    results = {}
    for test_file in test_files:
        passed = run_test(test_file)
        results[test_file.name] = passed
    
    # Print summary
    print(f"\n{'='*80}")
    print("TEST SUMMARY")
    print('='*80)
    
    passed_count = sum(1 for p in results.values() if p)
    total_count = len(results)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nResults: {passed_count}/{total_count} tests passed")
    
    all_passed = passed_count == total_count
    
    if all_passed:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  {total_count - passed_count} test(s) failed")
    
    return all_passed

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

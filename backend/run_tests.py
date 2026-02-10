#!/usr/bin/env python3
"""
Master test runner for all agent tests
Run with: python run_tests.py [--all | --inspector | --integration]
"""

import asyncio
import argparse
import sys
import os

def run_inspector_tests():
    """Run inspector agent tests"""
    print("🧪 Running Inspector Agent Tests...")
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    # Import and run test
    from tests.test_inspector import main as inspector_main
    return asyncio.run(inspector_main())

def run_integration_tests():
    """Run integration tests"""
    print("🔄 Running Integration Tests...")
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    from tests.test_integration import test_full_workflow
    return asyncio.run(test_full_workflow())

def run_all_tests():
    """Run all tests"""
    print("🚀 Running All Tests")
    print("="*80)
    
    # Run inspector tests
    print("\n" + "="*80)
    print("TEST PHASE 1: Inspector Agent")
    print("="*80)
    run_inspector_tests()
    
    # Run integration tests
    print("\n" + "="*80)
    print("TEST PHASE 2: Agent Integration")
    print("="*80)
    run_integration_tests()
    
    print("\n" + "="*80)
    print("🎉 ALL TESTS COMPLETED")
    print("="*80)

def run_diagnostician_tests():
    """Run diagnostician agent tests"""
    print("🔬 Running Diagnostician Agent Tests...")
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    # Import and run test
    from agents.diagnostician import test_diagnostician
    return asyncio.run(test_diagnostician())

def run_full_pipeline_tests():
    """Run full pipeline tests"""
    print("🚀 Running Full Pipeline Tests...")
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    from tests.test_full_pipeline import test_full_pipeline
    return asyncio.run(test_full_pipeline())

def run_fixer_tests():
    """Run fixer agent tests"""
    print("🔧 Running Fixer Agent Tests...")
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    from agents.fixer import test_fixer
    return asyncio.run(test_fixer())

def run_complete_pipeline_tests():
    """Run complete pipeline tests"""
    print("🚀 Running Complete Pipeline Tests...")
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    from tests.test_full_pipeline import test_complete_pipeline
    return asyncio.run(test_complete_pipeline())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run agent tests")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--inspector", action="store_true", help="Run inspector tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration tests only")
    parser.add_argument("--quick", action="store_true", help="Quick integration test")
    
    args = parser.parse_args()
    
    if args.all or (not args.inspector and not args.integration and not args.quick):
        run_all_tests()
    elif args.inspector:
        run_inspector_tests()
    elif args.integration:
        run_integration_tests()
    elif args.quick:
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from tests.test_integration import quick_test
        asyncio.run(quick_test())
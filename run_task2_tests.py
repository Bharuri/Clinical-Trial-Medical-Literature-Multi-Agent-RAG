#!/usr/bin/env python3
"""
Simple test discovery and execution for Task 2.
"""

import subprocess
import sys
from pathlib import Path


def run_tests():
    """Run all Task 2 tests."""
    test_files = [
        "tests/test_ingestion_chunking.py",
        "tests/test_ingestion_embeddings.py",
        "tests/test_ingestion_storage.py",
        "tests/test_ingestion_pdf_extraction.py",
    ]

    print("=" * 60)
    print("TASK 2 TEST EXECUTION")
    print("=" * 60)
    
    # Check if pytest is available
    try:
        result = subprocess.run(
            ["python", "-m", "pytest", "--version"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print("pytest not found. Install with: pip install -r requirements.txt")
            return 1
    except Exception as e:
        print(f"Error checking pytest: {e}")
        print("Install dependencies: pip install -r requirements.txt")
        return 1
    
    # Run tests
    cmd = ["python", "-m", "pytest", "-v", "--tb=short"] + test_files
    
    print(f"\nRunning: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd)
    
    return result.returncode


if __name__ == "__main__":
    sys.exit(run_tests())

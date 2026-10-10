#!/usr/bin/env python3
"""
Simple syntax validation script for Task 1.
Run this to verify all Python files compile without syntax errors.
"""

import ast
import sys
from pathlib import Path


def validate_python_file(file_path: Path) -> bool:
    """Validate that a Python file has correct syntax."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            ast.parse(f.read(), filename=str(file_path))
        return True
    except SyntaxError as e:
        print(f"❌ Syntax error in {file_path}: {e}")
        return False
    except Exception as e:
        print(f"❌ Error validating {file_path}: {e}")
        return False


def main() -> int:
    """Validate all Python files in the src directory."""
    root = Path(__file__).parent
    
    python_files = list(root.glob("src/**/*.py")) + list(root.glob("tests/**/*.py"))
    
    if not python_files:
        print("No Python files found to validate")
        return 1
    
    print(f"Validating {len(python_files)} Python files...\n")
    
    all_valid = True
    for py_file in sorted(python_files):
        if validate_python_file(py_file):
            rel_path = py_file.relative_to(root)
            print(f"✓ {rel_path}")
        else:
            all_valid = False
    
    print()
    if all_valid:
        print("✓ All Python files are valid!")
        return 0
    else:
        print("❌ Some files have syntax errors")
        return 1


if __name__ == "__main__":
    sys.exit(main())

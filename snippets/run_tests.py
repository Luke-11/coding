#!/usr/bin/env python3
"""
Test runner script for LaTeX Dependency Collector tests.
"""

import subprocess
import sys
from pathlib import Path

def run_tests(args=None):
    """Run pytest with given arguments."""
    if args is None:
        args = []

    cmd = [sys.executable, "-m", "pytest"] + args
    result = subprocess.run(cmd, cwd=Path(__file__).parent)

    return result.returncode

def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Run LaTeX Dependency Collector tests")
    parser.add_argument("--coverage", action="store_true", help="Run with coverage reporting")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--quick", action="store_true", help="Run only core tests (skip optional dependencies)")
    parser.add_argument("pytest_args", nargs="*", help="Additional pytest arguments")

    args = parser.parse_args()

    test_args = []

    if args.verbose:
        test_args.append("-v")

    if args.coverage:
        test_args.extend(["--cov=dependency_collector", "--cov-report=html", "--cov-report=term"])

    if args.quick:
        # Skip tests that require optional dependencies
        test_args.extend(["-k", "not (dataframe or networkx or plotly or graph)"])

    test_args.extend(args.pytest_args)

    if not test_args:
        test_args = ["-x", "--tb=short"]

    print(f"Running: pytest {' '.join(test_args)}")
    return run_tests(test_args)

if __name__ == "__main__":
    sys.exit(main())
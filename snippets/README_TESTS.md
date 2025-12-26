# LaTeX Dependency Collector Tests

This directory contains comprehensive pytest tests for the `LatexDependencyCollector` class.

## Running Tests

### Basic Test Run

```bash
pytest test_dependency_collector.py
```

### With Coverage

```bash
pytest test_dependency_collector.py --cov=dependency_collector --cov-report=html
```

### Specific Test

```bash
pytest test_dependency_collector.py::TestLatexDependencyCollector::test_collect_dependencies_main -v
```

## Test Structure

The test suite covers:

### Core Functionality

- **Initialization**: Basic setup and configuration
- **Path Normalization**: File path handling and LaTeX conventions
- **Dependency Parsing**: Regex pattern matching for LaTeX commands
- **Recursive Collection**: Building complete dependency trees
- **Data Structures**: Tree representation and file listing

### Optional Features (Skipped if dependencies not available)

- **DataFrame Export**: pandas integration
- **Graph Visualization**: NetworkX and Plotly integration
- **File Export**: HTML/JSON graph saving

### Edge Cases

- **Missing Files**: Handling non-existent dependencies
- **Circular Dependencies**: Prevention of infinite recursion
- **Empty Files**: Handling files with no dependencies
- **Invalid Paths**: Error handling for malformed inputs

## Test Fixtures

- `temp_dir`: Temporary directory for test files
- `sample_latex_files`: Complete LaTeX project structure with dependencies
- `collector`: Configured LatexDependencyCollector instance

## Dependencies

Core dependencies:

- pytest
- pytest-cov (optional, for coverage reporting)

Optional dependencies (for full test coverage):

- pandas
- networkx
- plotly

Install test dependencies:

```bash
pip install -r requirements-test.txt
```

## Test Coverage

The tests provide comprehensive coverage of:

- All public methods
- Error conditions
- Edge cases
- Integration with optional dependencies
- LaTeX parsing accuracy

Run with coverage to see detailed metrics:

```bash
pytest --cov=dependency_collector --cov-report=term-missing
```

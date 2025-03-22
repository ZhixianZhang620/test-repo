import pytest
import sys
import os

# Add project root directory to path so imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# This runs at import time before any tests
def setup_paths():
    # Get the project root directory
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # Add the project root to the Python path
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # Add specific directories needed for imports
    paths_to_add = [
        os.path.join(project_root, 'model_training', 'Server'),  
        os.path.join(project_root, 'inference_service'),
        # Add any other directories as needed
    ]
    
    for path in paths_to_add:
        if os.path.exists(path) and path not in sys.path:
            sys.path.insert(0, path)
            print(f"Added to Python path: {path}")
        elif not os.path.exists(path):
            print(f"Warning: Path does not exist: {path}")

# Call this function when conftest.py is imported
setup_paths()

# Define custom markers
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: mark a test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark a test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark a test as slow-running"
    )

# Skip slow tests by default, run with --run-slow to include them
def pytest_addoption(parser):
    parser.addoption(
        "--run-slow", action="store_true", default=False, help="run slow tests"
    )

def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-slow"):
        skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
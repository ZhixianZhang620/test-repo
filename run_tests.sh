#!/bin/bash
# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print section header
print_header() {
    echo -e "\n${BLUE}============================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}============================================================${NC}\n"
}

# Install test dependencies if needed
if [ "$1" == "--install" ]; then
    print_header "Installing test dependencies"
    pip install pytest pytest-cov coverage pytest-xdist
    exit 0
fi

# Create an empty __init__.py file in the tests directory if it doesn't exist
if [ ! -f "tests/__init__.py" ]; then
    touch tests/__init__.py
fi

# Default to running all tests if no arguments provided
if [ $# -eq 0 ]; then
    print_header "Running all tests"
    python -m pytest tests/ -v
    exit $?
fi

# Run unit tests
if [ "$1" == "--unit" ]; then
    print_header "Running unit tests"
    python -m pytest tests/ -v -m "unit"
    exit $?
fi

# Run integration tests
if [ "$1" == "--integration" ]; then
    print_header "Running integration tests"
    python -m pytest tests/ -v -m "integration"
    exit $?
fi

# Run specific test file
if [ "$1" == "--file" ] && [ -n "$2" ]; then
    print_header "Running tests in $2"
    python -m pytest tests/$2 -v
    exit $?
fi

# Run backend tests only
if [ "$1" == "--backend" ]; then
    print_header "Running backend tests"
    python -m pytest tests/test_backend_app.py -v
    exit $?
fi

# Run model tests only
if [ "$1" == "--model" ]; then
    print_header "Running model tests"
    python -m pytest tests/test_svd_model*.py tests/test_model_app.py -v
    exit $?
fi

# Run data processing tests only
if [ "$1" == "--data" ]; then
    print_header "Running data processing tests"
    python -m pytest tests/test_kafka_consumer*.py tests/data_process_test.py -v
    exit $?
fi

# Run slow integration tests
if [ "$1" == "--integration-slow" ]; then  # Changed this to avoid duplicate
    print_header "Running integration tests (including slow tests)"
    python -m pytest tests/ -v -m "integration" --run-slow
    exit $?
fi

# Run code coverage report
if [ "$1" == "--coverage" ]; then
    print_header "Running tests with coverage"
    coverage_report_dir="coverage_report"  # Changed back to project root directory
    mkdir -p $coverage_report_dir
    
    # Run pytest with coverage
    python -m pytest tests/ --cov=. --cov-report=html:$coverage_report_dir
    
    coverage_result=$?
    
    # Open coverage report if generated successfully
    if [ $coverage_result -eq 0 ]; then
        echo -e "\n${GREEN}Coverage report generated in $coverage_report_dir${NC}"
        echo -e "Open $coverage_report_dir/index.html in your browser to view the report"
    else
        echo -e "\n${RED}Error generating coverage report${NC}"
    fi
    
    exit $coverage_result
fi

# Help message
print_header "Test runner help"
echo "Usage:"
echo "  ./run_tests.sh             - Run all tests"
echo "  ./run_tests.sh --unit      - Run only unit tests"
echo "  ./run_tests.sh --integration - Run only integration tests"
echo "  ./run_tests.sh --integration-slow - Run integration tests including slow ones"
echo "  ./run_tests.sh --file FILE - Run tests in specific file (e.g. test_svd_model.py)"
echo "  ./run_tests.sh --backend   - Run backend API tests"
echo "  ./run_tests.sh --model     - Run model and ML service tests"
echo "  ./run_tests.sh --data      - Run data processing tests"
echo "  ./run_tests.sh --coverage  - Generate coverage report"
echo "  ./run_tests.sh --install   - Install test dependencies"
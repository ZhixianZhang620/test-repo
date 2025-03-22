# Movie Recommendation System Tests

This directory contains comprehensive tests for the movie recommendation system. The tests cover all components of the system, including data processing, model training, and API services.

## Test Structure

* **Unit Tests** : Test individual components in isolation
* **Integration Tests** : Test how components work together
* **End-to-End Tests** : Test the complete system flow

## Test Files

### Data Processing Tests

* `test_kafka_consumer_comprehensive.py`: Tests for the Kafka consumer component
* `data_process_test.py`: Tests for data processing functions

### Model Tests

* `test_svd_model.py`: Tests for the SVD recommendation model
* `test_svd_model_extended.py`: Additional tests for edge cases and error handling

### API Service Tests

* `test_model_app.py`: Tests for the ML model service API
* `test_backend_app.py`: Tests for the backend recommendation API
* `test_e2e_integration.py`: End-to-end tests for the entire system

### Test Configuration

* `conftest.py`: Pytest configuration and shared fixtures
* `pytest.ini`: Pytest settings and marker definitions

## Running Tests

Use the provided `run_tests.sh` script to run the tests:

```bash
# Install test dependencies
./run_tests.sh --install

# Run all tests
./run_tests.sh

# Run only unit tests
./run_tests.sh --unit

# Run specific components
./run_tests.sh --backend  # Backend API tests
./run_tests.sh --model    # Model and ML service tests
./run_tests.sh --data     # Data processing tests

# Run integration tests (including slow tests)
./run_tests.sh --integration

# Run a specific test file
./run_tests.sh --file test_svd_model.py

# Generate coverage report
./run_tests.sh --coverage
```

## Test Coverage

The tests aim to provide comprehensive coverage of:

* Normal operation paths
* Error handling
* Edge cases
* Input validation
* Database interactions
* API functionality

## Adding New Tests

When adding new tests:

1. Follow the naming convention: `test_*.py` for files, `test_*` for functions
2. Add appropriate markers (`@pytest.mark.unit`, `@pytest.mark.integration`, etc.)
3. Use fixtures from `conftest.py` when applicable
4. Mock external dependencies to avoid network/database access
5. Consider both success and failure scenarios

## Best Practices

* Keep tests independent (no dependencies between tests)
* Clean up resources after tests
* Use descriptive test names that explain what's being tested
* Test both expected and error conditions
* Minimize hardcoded test data

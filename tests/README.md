# Tests

This directory contains tests for the home-energy-planner functionality.

## Structure

- `conftest.py` - Pytest configuration and fixtures
- `test_planner.py` - Main test suite for planner functionality
- `generate_fixtures.py` - Script to generate sample XLSX fixture files
- `fixtures/` - Directory containing sample XLSX files for testing

## Running Tests

Run all tests:
```bash
poetry run pytest tests/
```

Run with verbose output:
```bash
poetry run pytest tests/ -v
```

Run a specific test file:
```bash
poetry run pytest tests/test_planner.py
```

Run a specific test:
```bash
poetry run pytest tests/test_planner.py::TestOTELoading::test_load_ote_prices_from_local_file
```

## Fixtures

The test fixtures include:

- **OTE XLSX files**: Sample day-ahead price data in the format expected by `load_ote_15min_prices_eur_per_mwh()`
- **CNB XLSX files**: Sample exchange rate data in the format expected by `load_cnb_eur_czk_rate()`

Fixtures are automatically generated if they don't exist when tests run. You can also generate them manually:

```bash
cd tests
python generate_fixtures.py
```

## Test Coverage

The test suite covers:

- Loading OTE 15-minute price data from XLSX files
- Loading CNB exchange rate data from XLSX files
- Decision computation logic (when charging is needed)
- Full planning workflow with local files
- Window preference settings
- Edge cases (no charging needed, etc.)

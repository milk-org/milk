# About tests

## About pytest in general

A few cli handfuls for future ref

```bash
# Run the full suite:
pytest
# A subfolder or a single files
pytest <glob_pattern>
# Test function name pattern:
pytest -k pattern
# Start postmortem on error:
pytest --pdb
# Verbose:
pytest -v
# stdin/stdout
pytest -s
```

## In this suite:

#### Global files:

The main files is the `conftest.py`

It includes auxiliary global files located in the `conftestaux` folder by means of:
```python
pytest_plugins = ["tests.conftestaux.milk", "tests.conftestaux.cacao_loop"]
```

#### Main test entrypoints

Some test environments can be set-up and held alive until user-terminated.
These environments are located _as tests_ that are normally excluded in the `tests.main` subpackage.
To the difference of normal tests, the execution flow holds upon an `input` statement, which allows:
- Start the test
- Test sets up, hangs on `input()`
- Perform testing in another terminal
- Press a key in test terminal to clear wait on `input()`
- test teardowns and pytest terminates.

This is how you would run one such test:
```bash
pytest -s tests/mains/servers.py
```
add an additional `-k` flag if these files end up contaning multiple tests.

## Python coverage

coverage is configure in the pyproject.toml to run the test suite and analyze code coverage.

```bash
coverage run
coverage report
```

## Coverage for C/C++ Extensions

With a special build, we can enable coverage in C files while running the test stack purely from pytest.

Compiling/installing with PIP:
```bash
COVERAGE=ON python setup.py build_ext --inplace --build-temp ./_build_pip
```

FIXME: we'll fix the following by disambiguating the libraries names.
```bash
cp _build_pip/pyMilk/*.so pyMilk/
sudo cp _build_pip/pyMilk/lib*.so /usr/local/milk/lib/ # Override milk libImageStreamIO.so... otherwise linking is confused.
coverage run
gcovr --verbose -r . --html-details -o gcov_html/c_coverage.html
firefox ./gcov_html/c_coverage.html
```

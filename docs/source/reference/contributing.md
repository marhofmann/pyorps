# 💻 Contributing

## Development Setup

```bash
# Clone and set up
git clone https://github.com/marhofmann/pyorps.git
cd pyorps
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e .[dev,full]
python setup.py build_ext --inplace
```

## Running Tests

```bash
pytest tests/ -v
pytest --cov=pyorps
```

## Code Style

PYORPS follows PEP 8 with a maximum line length of 88 (Black convention).

```bash
black pyorps/
isort pyorps/
flake8 pyorps/
```

## Commit Format

Use conventional commits:

- `feat:` -- new feature
- `fix:` -- bug fix
- `docs:` -- documentation only
- `refactor:` -- code restructuring without behavior change
- `test:` -- adding or updating tests
- `chore:` -- maintenance tasks

## Pull Request Checklist

- Code follows the style guidelines above
- All tests pass locally
- Cython extensions build successfully
- Documentation updated if needed
- New tests added for new features

## Cython Contributions

After modifying any `.pyx` file, rebuild the extensions before testing:

```bash
python setup.py build_ext --inplace
```

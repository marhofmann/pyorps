# Contributing to PYORPS

We welcome contributions to PYORPS! Whether you've found a bug, have a suggestion for a new feature, or want to contribute code, your input is highly valued. PYORPS now includes high-performance Cython extensions for optimal pathfinding algorithms, making contributions even more impactful.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Environment Setup](#development-environment-setup)
- [Building Cython Extensions](#building-cython-extensions)
- [Making Changes](#making-changes)
- [Testing Your Changes](#testing-your-changes)
- [Pull Request Process](#pull-request-process)
- [Code Style Guidelines](#code-style-guidelines)
- [Contributing to Cython Code](#contributing-to-cython-code)
- [Reporting Issues](#reporting-issues)
- [Suggesting Enhancements](#suggesting-enhancements)
- [Release Process](#release-process)
- [Recognition](#recognition)

## Getting Started

### 🚀 Ways to Contribute

- **Code contributions**: Bug fixes, new features, performance improvements
- **Cython optimization**: Improve existing algorithms or add new high-performance implementations
- **Documentation**: README updates, code comments, tutorials
- **Testing**: Add test cases, improve test coverage
- **Examples**: Contribute case studies or example scripts
- **Bug reports**: Help us identify and fix issues
- **Feature requests**: Suggest new functionality

### 📞 Get in Touch

- **Issues**: Open an issue on the [PYORPS GitHub issue board](https://github.com/marhofmann/pyorps/issues)
- **Discussions**: Use GitHub Discussions for questions and ideas
- **Email**: Contact the maintainer at martin.hofmann-3@ei.thm.de

## Development Environment Setup

### Prerequisites

- Python 3.12 or higher
- Git
- C++ compiler (MSVC on Windows, GCC/Clang on Linux/macOS)
- GitHub account

### Required Build Tools

```bash
# Install build dependencies
pip install --upgrade pip setuptools wheel
pip install cython>=3.0.0 numpy>=2.0.0
```

### Setup Instructions

1. **Fork and Clone**
   ```bash
   # Fork the repository on GitHub, then clone your fork
   git clone https://github.com/YOUR-USERNAME/pyorps.git
   cd pyorps
   
   # Add upstream remote
   git remote add upstream https://github.com/marhofmann/pyorps.git
   ```

2. **Create Virtual Environment**
   ```bash
   # Create and activate virtual environment
   python -m venv .venv
   
   # Windows
   .venv\Scripts\activate
   
   # Linux/macOS
   source .venv/bin/activate
   ```

3. **Install Development Dependencies**
   ```bash
   # Install package in development mode with all dependencies
   pip install -e .[dev,full]
   ```

4. **Verify Installation**
   ```bash
   # Run tests to ensure everything works
   pytest tests/ -v
   ```

## Building Cython Extensions

PYORPS includes Cython extensions for high-performance pathfinding algorithms. Here's how to work with them:

### Building for Development

```bash
# Build Cython extensions in-place for development
python setup.py build_ext --inplace

# Or use the provided script
python scripts/build_cython.py
```

### Platform-Specific Notes

**Windows:**
- Requires Visual Studio Build Tools or Visual Studio
- May need to install Windows SDK

**Linux:**
- Requires GCC with C++ support: `sudo apt-get install build-essential`

**macOS:**
- Requires Xcode command line tools: `xcode-select --install`

### Verifying Cython Build

```bash
# Test that Cython extensions are working
python -c "
from pyorps.utils.find_path_cython import dijkstra_2d_cython
print('✓ Cython extensions built successfully')
"
```

## Making Changes

### 1. Create a Feature Branch

```bash
# Always create a new branch for your changes
git checkout -b feature/descriptive-name
# or
git checkout -b fix/issue-number
```

### 2. Development Workflow

- **Pure Python changes**: Edit files directly and test
- **Cython changes**: Edit `.pyx` files, rebuild extensions, then test
- **Documentation**: Update docstrings, README, or documentation files

### 3. Commit Your Changes

```bash
# Stage your changes
git add .

# Commit with a descriptive message
git commit -m "feat: add new pathfinding algorithm

- Implements A* algorithm in Cython
- Includes comprehensive tests
- Updates documentation
"
```

## Testing Your Changes

### Running Tests

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=pyorps

# Run specific test file
pytest tests/test_pathfinding.py

# Run tests for Cython extensions specifically
pytest tests/test_cython_extensions.py -v
```

### Testing Cython Changes

```bash
# After modifying .pyx files, rebuild and test
python setup.py build_ext --inplace
pytest tests/test_cython_extensions.py

# Test performance improvements
python scripts/benchmark_cython.py
```

### Building Wheels Locally

```bash
# Build wheel for testing
python -m build

# Test the built wheel
pip install dist/pyorps-*.whl --force-reinstall
python -c "from pyorps.utils.find_path_cython import dijkstra_2d_cython; print('Success!')"
```

## Pull Request Process

### Before Submitting

- [ ] Code follows project style guidelines
- [ ] All tests pass locally
- [ ] Cython extensions build successfully
- [ ] Documentation is updated (if applicable)
- [ ] New tests added for new functionality
- [ ] Performance benchmarks run (for Cython changes)

### Pull Request Checklist

1. **Create Pull Request**
   ```bash
   # Push your branch
   git push origin feature/your-feature-name
   ```
   
2. **PR Description Should Include:**
   - Clear description of changes
   - Issue number (if applicable): `Fixes #123`
   - Breaking changes (if any)
   - Performance impact (for Cython changes)

3. **Automated Checks**
   - All tests pass
   - Wheels build successfully on all platforms
   - Code style checks pass
   - Documentation builds correctly

### Review Process

- Maintainers will review your PR within 1-2 weeks
- Address feedback by pushing new commits to your branch
- Once approved, maintainers will merge your PR

## Code Style Guidelines

### Python Code

- Follow [PEP 8](https://peps.python.org/pep-0008/) style guidelines
- Use type hints where appropriate
- Add docstrings to all public functions and classes
- Maximum line length: 88 characters (Black formatter)

```bash
# Format code
black pyorps/
isort pyorps/

# Check style
flake8 pyorps/
```

### Cython Code

- Use `.pyx` extension for Cython files
- Follow Python naming conventions
- Add type declarations for performance-critical code
- Include comprehensive docstrings

```cython
# Example Cython function
cpdef double dijkstra_2d_cython(
    double[:, :] cost_matrix,
    tuple start,
    tuple end
):
    """
    High-performance Dijkstra pathfinding algorithm.
    
    Parameters
    ----------
    cost_matrix : double[:, :]
        2D cost matrix
    start : tuple
        Starting coordinates (row, col)
    end : tuple
        Target coordinates (row, col)
        
    Returns
    -------
    double
        Total path cost
    """
    # Implementation here
```

### Commit Messages

Use conventional commit format:
```
type(scope): description

[optional body]

[optional footer]
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

## Contributing to Cython Code

### Understanding the Cython Extensions

The main Cython module is `pyorps/utils/find_path_cython.pyx` which contains:
- `dijkstra_2d_cython`: High-performance 2D Dijkstra implementation
- `dijkstra_single_source_multiple_targets`: Optimized multi-target pathfinding
- `create_exclude_mask`: Fast exclusion mask generation

### Performance Considerations

- Use memory views for NumPy arrays: `double[:, :] array`
- Declare variables with `cdef` for better performance
- Use `cpdef` for functions that need both Python and Cython access
- Profile your changes with `cProfile` and `line_profiler`

### Benchmarking

```bash
# Run performance benchmarks
python scripts/benchmark_pathfinding.py

# Compare before and after your changes
python scripts/compare_performance.py
```

## Reporting Issues

### Bug Reports

Use the [bug report template](https://github.com/marhofmann/pyorps/issues/new?template=bug_report.md) and include:
- Python version and operating system
- PYORPS version
- Complete error traceback
- Minimal code example to reproduce
- Expected vs. actual behavior

### Security Issues

For security-related issues, email martin.hofmann-3@ei.thm.de directly instead of opening a public issue.

## Suggesting Enhancements

Use the [enhancement template](https://github.com/marhofmann/pyorps/issues/new?template=enhancement.md) and include:
- Clear description of the proposed feature
- Use case and motivation
- Possible implementation approach
- Performance considerations (if applicable)

## Release Process

### Version Numbering

- Follow [Semantic Versioning](https://semver.org/)
- Major: Breaking changes
- Minor: New features, backward compatible (a change of reported numbers, such as lengths in CRS units, counts as breaking)
- Patch: Bug fixes, backward compatible
- Release candidates: `X.Y.ZrcN` (PEP 440), for example `0.5.0rc1`

### What runs automatically

| Workflow | When | What it checks |
|---|---|---|
| **Code Quality Check** | pull requests and pushes to `main` and `develop` | flake8 and Pylint (blocking, threshold in the workflow) |
| **Tests** | pull requests and pushes to `main` and `develop` | the whole test suite from a clean checkout, Linux (3.12, 3.13) and Windows (3.12) |
| **Tests with the newest dependencies** | every Monday and on demand | the suite against the newest numpy, scipy, rasterio, affine, ... |
| **Build and Publish to PyPI** | a tag `v*` (or a manual dry run) | release gate, wheels for Linux, Windows and macOS, wheel tests, publish |

The release gate refuses a tag whose version differs from `pyproject.toml` or `pyorps/__init__.py`, and a tag on a
commit that is not on `main`.

### Releasing, step by step

1. **Everything is merged to `main` first** (squash merge from `develop`), and *Tests* and *Code Quality Check* are green.
2. **Bump the version** in `pyproject.toml` and `pyorps/__init__.py`, update the changelog and the status labels of the
   documentation pages (`docs/source/getting_started/release_status.md`), and merge that too.
3. **Check what git tracks, not what your machine has.** Run, from the repository root:

   ```bash
   python tools/release_check.py            # tests of the committed tree, exactly as the wheel-test job runs them
   python tools/release_check.py --build    # additionally builds the wheel and tests it in a clean virtual environment
   ```

   Files that are ignored or untracked (a `.gitignore` entry for `profiles/` hid four required files for months) do
   not exist in this check, so they cannot hide a failure.
4. **Rehearse with a release candidate.** Push `v0.5.0rc1` on the commit from step 2. The pipeline builds, tests and
   publishes to **TestPyPI only**. Install it in a clean environment and look at it.
5. **Push the final tag last**: `git tag v0.5.0 && git push origin v0.5.0`. The commit must already be on `main`.
6. **Verify** with `pip install --upgrade pyorps` in a clean environment and check `pyorps.__version__`.

### If a release run fails

- A failure of the infrastructure (a download answering 503, a runner without capacity): re-run the failed jobs from
  the Actions page. **Do not delete or move the tag.**
- A failure in the code, the tests or the packaging: fix it on `main`, then publish the next patch version or the next
  release candidate (`0.5.0rc2`). PyPI never lets a version number be reused, and a moved tag makes the history
  untrustworthy.
- A tag that was pushed before the fix was merged is the most common mistake. The release gate catches the version
  mismatch and the missing merge.

### One-time setup (maintainers)

- PyPI: a *trusted publisher* for this repository and the environment `pypi`.
- TestPyPI (release candidates): a trusted publisher on <https://test.pypi.org> and a GitHub environment `testpypi`.

### Multi-Platform Wheels

The release builds wheels for Windows (x64), macOS (Apple Silicon) and Linux (x64) for the CPython versions listed in `pyproject.toml` (the build matrix in the workflow follows it).
The wheel tests install each wheel with the `full` extra in a clean environment and run the tests from a temporary
folder, so only files that git tracks are available to them.

### Writing tests that survive a clean machine

- A test may only read files that are tracked in git. Shared helpers belong in `tests/`; the CI job copies only
  `tests/` and `profiles/`.
- Tests that need a GPU, the GUI packages or the source tree must skip themselves (`@gpu_only`, `pytest.importorskip`
  inside the test module, `collect_ignore_glob` in a `conftest.py` - never a module-level `importorskip` in a
  `conftest.py`, it cancels the whole session). `tests/test_ci_hygiene.py` checks the common mistakes.
- Do not assume the first warning a call emits is yours: pick it by category (`ThinForbiddenFeatureWarning`, ...).
- A deprecation warning raised from pyorps code fails the test run (see `filterwarnings` in `pyproject.toml`); fix the
  call instead of silencing it.

## Recognition

### Contributors

We recognize contributors in multiple ways:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- GitHub contributor statistics
- Academic citations in research papers

### Thank You! 🙏

Your contributions make PYORPS better for the entire power systems community. Whether you're fixing bugs, adding features, or improving documentation, every contribution matters.

---

**Questions?** Don't hesitate to ask! Open an issue or start a discussion on GitHub.
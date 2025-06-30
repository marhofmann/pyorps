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

- Python 3.11 or higher
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
- Minor: New features, backward compatible
- Patch: Bug fixes, backward compatible

### Multi-Platform Wheels

The project uses GitHub Actions to automatically build wheels for:
- Windows (x64)
- macOS (Intel + Apple Silicon)
- Linux (x64 + ARM64)
- Python 3.11, 3.12+

### Release Checklist

1. Update version in `pyproject.toml`
2. Update CHANGELOG.md
3. Create and push tag: `git tag v0.x.x && git push origin v0.x.x`
4. GitHub Actions automatically builds and publishes wheels

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
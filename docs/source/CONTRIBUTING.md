## Contributing to PYORPS

We welcome contributions to PYORPS! Whether you’ve found a bug, have a suggestion for a new feature, or want to contribute code, your input is highly valued. Please follow the guidelines below to get started.

### Get in Touch!

If you’ve found a bug or have an idea for a new feature, open an issue on the [PYORPS GitHub issue board](https://github.com/marhofmann/pyorps/issues). This is the best way to discuss potential improvements with the community and maintainers.

### Setting Up Your Development Environment

If you’d like to contribute code, follow these steps to set up your development environment:

1. **Install Git and Create a GitHub Account**  
   If you haven’t already, install Git and create a GitHub account. See [GitHub’s guide](https://docs.github.com/en/get-started/quickstart/set-up-git) for help.

2. **Fork the Repository**  
   Go to the [official PYORPS repository](https://github.com/marhofmann/pyorps) and click the "Fork" button to create your own copy of the repository.

3. **Clone Your Fork**  
   Clone your forked repository to your local machine:
   ```bash
   git clone https://github.com/YOUR-USERNAME/pyorps.git
   ```

4. **Set Up Remotes**  
   Add the official repository as an upstream remote to keep your fork in sync:
   ```bash
   cd pyorps
   git remote add upstream https://github.com/marhofmann/pyorps.git
   ```

5. **Install Dependencies**  
   Install the required dependencies for development:
   ```bash
   pip install -e .[dev]
   ```

6. **Run Tests**  
   Ensure everything is working by running the test suite:
   ```bash
   pytest
   ```

### Making Changes

1. **Create a Branch**  
   Always create a new branch for your changes. Use a descriptive name for the branch:
   ```bash
   git checkout -b feature/my-new-feature
   ```

2. **Make Your Changes**  
   Edit the code and make your changes. Be sure to follow the project’s coding style and add comments where necessary.

3. **Add and Commit Your Changes**  
   Stage and commit your changes with a meaningful commit message:
   ```bash
   git add .
   git commit -m "Add feature: my new feature"
   ```

4. **Push Your Changes**  
   Push your branch to your fork:
   ```bash
   git push origin feature/my-new-feature
   ```

5. **Submit a Pull Request**  
   Go to the [official PYORPS repository](https://github.com/marhofmann/pyorps) and open a pull request from your branch. Provide a clear description of your changes and reference any related issues.

6. **Respond to Feedback**  
   If the maintainers or community members request changes, update your branch and push the changes. The pull request will automatically update.

### Keeping Your Fork Up-to-Date

To keep your fork in sync with the official repository, regularly pull changes from the upstream repository:
```bash
git checkout develop
git pull upstream develop
git push origin develop
```

### Writing Tests

If you add new functionality, please include tests to ensure it works as expected. PYORPS uses `pytest` for testing. Add your tests to the `tests` directory, following the existing structure. For example:

```python
def test_new_feature():
    # Your test code here
    assert True
```

Run the tests locally before submitting your pull request:
```bash
pytest
```

### Code Style

Please follow the [PEP 8](https://peps.python.org/pep-0008/) coding style guide. Use tools like `flake8` or `black` to check and format your code:
```bash
pip install flake8 black
flake8 pyorps
black pyorps
```

### Contributing Examples or Case Studies

If you’d like to contribute example scripts or case studies, add them to the `examples` or `case_studies` directories, respectively. Be sure to include clear comments and documentation to help others understand your work.

### Thank You!

Your contributions make PYORPS better for everyone. Thank you for taking the time to contribute!


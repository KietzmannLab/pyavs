# Contributing to pyAVS

We welcome contributions to pyAVS! This guide will help you get started.

## Quick Start

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/pyavs.git
   cd pyavs
   ```
3. Set up development environment:
   ```bash
   pip install -e ".[dev]"
   pre-commit install
   ```
4. Create a branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Workflow

### Code Style

We use several tools to maintain code quality:

- **Black**: Code formatting
- **Flake8**: Linting  
- **isort**: Import sorting

Run these before committing:
```bash
black pyavs/
flake8 pyavs/  
isort pyavs/
```

### Testing

Run the test suite:
```bash
pytest tests/

# With coverage
pytest --cov=pyavs tests/
```

### Documentation

Build documentation locally:
```bash
cd docs/
make html

# View in browser
open _build/html/index.html
```

## Types of Contributions

### Bug Reports
When reporting bugs, please include:
- Operating system and version
- Python version
- pyAVS version
- Minimal example to reproduce the issue
- Full error traceback

### Feature Requests
Before suggesting new features:
- Check if it's already been requested
- Explain the use case clearly
- Consider if it fits pyAVS's scope

### Code Contributions
Good first issues:
- Documentation improvements
- Adding examples
- Fixing bugs
- Adding tests

Major contributions should be discussed in an issue first.

## Pull Request Process

1. Ensure your code follows the style guidelines
2. Add or update tests as needed
3. Update documentation if necessary
4. Ensure all tests pass
5. Create a pull request with:
   - Clear description of changes
   - Link to related issues
   - Screenshots if UI changes

## Community Guidelines

### Be Respectful
- Use welcoming and inclusive language
- Be respectful of different viewpoints
- Accept constructive criticism gracefully

### Be Helpful
- Help newcomers get started
- Share knowledge and experience
- Provide constructive feedback

## Getting Help

If you need help:
- Check existing documentation
- Look at similar implementations in the codebase
- Ask questions in GitHub discussions
- Contact maintainers directly

Thank you for contributing to pyAVS! 🧠✨
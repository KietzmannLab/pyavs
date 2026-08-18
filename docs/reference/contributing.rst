Contributing to pyAVS
=====================

We welcome contributions to pyAVS! This guide will help you get started.

Getting Started
---------------

1. Fork the repository on GitHub
2. Clone your fork locally:

.. code-block:: bash

   git clone https://github.com/your-username/pyavs.git
   cd pyavs

3. Set up development environment:

.. code-block:: bash

   pip install -e ".[dev]"
   pre-commit install

4. Create a branch for your feature:

.. code-block:: bash

   git checkout -b feature/your-feature-name

Development Workflow
--------------------

Code Style
~~~~~~~~~~

We use several tools to maintain code quality:

- **Black**: Code formatting
- **Flake8**: Linting
- **isort**: Import sorting

Run these before committing:

.. code-block:: bash

   black pyavs/
   flake8 pyavs/
   isort pyavs/

Testing
~~~~~~~

Run the test suite:

.. code-block:: bash

   pytest tests/
   
   # With coverage
   pytest --cov=pyavs tests/

Documentation
~~~~~~~~~~~~~

Build documentation locally:

.. code-block:: bash

   cd docs/
   make html
   
   # View in browser
   open _build/html/index.html

Types of Contributions
----------------------

Bug Reports
~~~~~~~~~~~

When reporting bugs, please include:

- Operating system and version
- Python version
- pyAVS version
- Minimal example to reproduce the issue
- Full error traceback

Feature Requests
~~~~~~~~~~~~~~~~

Before suggesting new features:

- Check if it's already been requested
- Explain the use case clearly
- Consider if it fits pyAVS's scope

Code Contributions
~~~~~~~~~~~~~~~~~~

Good first issues:

- Documentation improvements
- Adding examples
- Fixing bugs
- Adding tests

Major contributions should be discussed in an issue first.

Contribution Guidelines
-----------------------

Code Standards
~~~~~~~~~~~~~~

- Follow PEP 8 style guidelines
- Use descriptive variable and function names
- Add docstrings to all public functions
- Include type hints where appropriate

.. code-block:: python

   def load_data(subject_id: int, session: int, 
                 verbose: bool = True) -> pd.DataFrame:
       """
       Load data for a specific subject and session.
       
       Parameters
       ----------
       subject_id : int
           Subject identifier
       session : int  
           Session number
       verbose : bool, optional
           Whether to print progress (default: True)
           
       Returns
       -------
       pd.DataFrame
           Loaded data
       """
       # Implementation here
       pass

Documentation Standards
~~~~~~~~~~~~~~~~~~~~~~~

- Use NumPy-style docstrings
- Include examples in docstrings when helpful
- Update relevant documentation when adding features
- Ensure all public APIs are documented

Testing Standards
~~~~~~~~~~~~~~~~~

- Write tests for all new functionality
- Aim for high test coverage
- Use pytest fixtures for setup
- Mock external dependencies

.. code-block:: python

   import pytest
   import pandas as pd
   from pyavs.dataloader import load_eye_events

   def test_load_eye_events():
       # Test with valid inputs
       result = load_eye_events(subject_id=1, session=1)
       assert isinstance(result, pd.DataFrame)
       assert len(result) > 0
       
   def test_load_eye_events_invalid_subject():
       # Test error handling
       with pytest.raises(ValueError):
           load_eye_events(subject_id=-1, session=1)

Pull Request Process
--------------------

1. Ensure your code follows the style guidelines
2. Add or update tests as needed
3. Update documentation if necessary
4. Ensure all tests pass
5. Create a pull request with:
   - Clear description of changes
   - Link to related issues
   - Screenshots if UI changes

Review Process
--------------

All contributions are reviewed by maintainers:

- Code quality and style
- Test coverage
- Documentation completeness
- Compatibility with existing code

Community Guidelines
--------------------

Be Respectful
~~~~~~~~~~~~~

- Use welcoming and inclusive language
- Be respectful of different viewpoints
- Accept constructive criticism gracefully

Be Helpful
~~~~~~~~~~

- Help newcomers get started
- Share knowledge and experience
- Provide constructive feedback

Release Process
---------------

pyAVS follows semantic versioning:

- **Major** (1.0.0): Breaking changes
- **Minor** (0.1.0): New features, backwards compatible  
- **Patch** (0.0.1): Bug fixes, backwards compatible

Development Setup Details
-------------------------

Environment Setup
~~~~~~~~~~~~~~~~~

Create a conda environment:

.. code-block:: bash

   conda create -n pyavs-dev python=3.9
   conda activate pyavs-dev
   pip install -e ".[dev,full]"

Pre-commit Hooks
~~~~~~~~~~~~~~~~

We use pre-commit to run checks automatically:

.. code-block:: bash

   pre-commit install
   
   # Run manually
   pre-commit run --all-files

IDE Setup
~~~~~~~~~

Recommended VS Code extensions:

- Python
- Pylance  
- Black Formatter
- Flake8
- autoDocstring

Getting Help
------------

If you need help:

- Check existing documentation
- Look at similar implementations in the codebase
- Ask questions in GitHub discussions
- Contact maintainers directly

Recognition
-----------

Contributors are recognized in:

- CHANGELOG.md for each release
- README.md contributors section
- Documentation acknowledgments

Thank you for contributing to pyAVS! 🧠✨
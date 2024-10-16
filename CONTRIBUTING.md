Contributing to bdi-kit
=======================

There are many ways to contribute to bdi-kit, such as improving the codebase, reporting 
issues or bugs, enhancing the documentation, reviewing pull requests from other developers, 
adding new matching methods, or expanding support for additional standards. 
See the instructions below to get started!


Adding New Matching Methods
---------------------------

Contributors can add new methods for schema and value matching by following these steps:

1. Create a Python module inside the `algorithms` folder (e.g., `bdikit/value_matching/algorithms`).

2. Define a class in the module that implements either `BaseValueMatcher` (for value matching) or `BaseSchemaMatcher` (for schema matching).

3. Instantiate an object of your class in `matcher_factory.py` (e.g., `bdikit/value_matching/matcher_factory.py`). Ensure your module is properly imported in the `__init__.py` file (e.g.,` bdikit/value_matching/__init__.py`).


Code of Conduct
---------------

We abide by the principles of openness, respect, and consideration of others
of the Python Software Foundation: https://www.python.org/psf/codeofconduct/.

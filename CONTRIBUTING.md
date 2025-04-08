Contributing to bdi-kit
=======================

There are many ways to contribute to bdi-kit, such as improving the codebase, reporting 
issues or bugs, enhancing the documentation, reviewing pull requests from other developers, 
adding new matching methods, or expanding support for additional standards. 
See the instructions below to get started!


Formatting the Code
-------------------

We format code using the [black](https://black.readthedocs.io/en/stable/) code formatter.
The CI runs for every pull request and will fail if code is not properly formatted.
To make sure formatting is correct, you can do the following steps.

Make sure you have black installed:
```
pip install black
```

To format the code, anyone can use the command before committing your changes:
```
make format
```

Or you can use the black command directly:
```
black ./bdikit/ ./tests/ ./scripts/
```


Adding New Matching Methods
---------------------------

Contributors can add new methods for schema and value matching by following these steps:

1. Create a Python module inside the "task folder" folder (e.g., `bdikit/value_matching`).

2. Define a class in the module that implements a base class. For value matching, it could be `BaseValueMatcher` or `BaseTopkValueMatcher`.  For schema matching, it could be `BaseSchemaMatcher` or `BaseTopkSchemaMatcher`.

3. Add a new entry to the Enum class (e.g. `ValueMatchers`) in `matcher_factory.py` (e.g., `bdikit/value_matching/matcher_factory.py`). 
Make sure to add the correct import path for your module to ensure it can be accessed without errors.


Adding New Standards
--------------------

Contributors can extend bdi-kit to additional standards  a by following these steps:

1. Create a Python module inside the "standards" folder (`bdikit/standards`).

2. Define a class in the module that implements `BaseStandard`.

3. Add a new entry to the class `Standards(Enum)` in `bdikit/standards/standard_factory.py`. Make sure to add the correct import path for your 
module to ensure it can be accessed without errors.


Code of Conduct
---------------

We abide by the principles of openness, respect, and consideration of others
of the Python Software Foundation: https://www.python.org/psf/codeofconduct/.
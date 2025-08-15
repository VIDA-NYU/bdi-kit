
Contributing
============

There are many ways to contribute to bdi-kit, such as improving the codebase, reporting 
issues or bugs, enhancing the documentation, reviewing pull requests from other developers, 
adding new matching methods, or expanding support for additional standards (data models). 
See the instructions below to get started!


Formatting the Code
-------------------

We format code using `black <https://black.readthedocs.io/>`__.
The CI runs for every pull request and will fail if code is not properly formatted.
To make sure formatting is correct, you can do the following steps.

Make sure you have black installed:

::

   $ pip install black


To format the code, anyone can use the command before committing your changes:

::

   $ make format


Or you can use the black command directly:

::

   $ black ./bdikit/ ./tests/ ./scripts/



Adding New Matching Methods
---------------------------

Contributors can add new methods for schema and value matching by following these steps:

1. Create a Python module inside the "task folder" folder (e.g., `bdikit/schema_matching`). 
2. Define a class in the module to implements a base class. For schema matching, the base class could be `BaseSchemaMatcher`
   or `BaseTopkSchemaMatcher`. For them, you need to implement the methods `match_schema()` or `rank_schema_matches()`, respectively.
   For value matching, the base class could be `BaseValueMatcher` or `BaseTopkValueMatcher`. For them, you need to implement the methods
   `match_value()` or `rank_value_matches()`, respectively.
3. Add a new entry to the Enum class (e.g. `SchemaMatchers`) in `matcher_factory.py` (e.g., `bdikit/schema_matching/matcher_factory.py`). Make sure to add the correct import path for your module to ensure it can be accessed without errors.


Adding New Standards (Data Models)
----------------------------------

Contributors can extend bdi-kit to additional standards (data models) by following these steps:

1. Create a Python module inside the "standards" folder (`bdikit/standards`). 
2. Define a class in the module to implements `BaseStandard`. This class should implement four methods:

   - `get_columns()`: Returns a list of all the columns (strings) of the standard.
   - `get_column_values()`: Returns a dictionary where the keys are column names and the values are lists of possible values for each column.
   - `get_column_metadata()`: Returns a dictionary where the keys are column names and the values are dictionaries containing these fields for each column:
     `column_description`, `value_names`, and `value_descriptions`.
   - `get_dataframe_rep()`: Returns a Pandas DataFrame representation of the standard, where each column in the DataFrame is a column in the standard and each row is a possible value for that column.

3. Add a new entry to the class `Standards(Enum)` in `bdikit/standards/standard_factory.py`. Make sure to add the correct import path for your module to ensure it can be accessed without errors.

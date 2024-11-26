Change Log
==========

0.4.0.dev0 (yyyy-mm-dd)
------------------------

We are pleased to announce the release of bdikit version 0.4.0.
This version includes some new features and breaking changes,
including a change in the minimum Python version supported (3.9)
and support to search top-k best-value matches for a given pair
of table attributes.

Below is a list of the main changes included in this release:

- chore: Update instructions
- fix: Check whether the values are numeric
- fix: Add supported versions of Python
- chore: remove bdi-viz (#84)
- feat: Support for top-k value matches
- chore: ensure nltk version is >=3.9.1 to fix a security issue
- chore: Remove unnecessary transitive dependency: conllu<5.0.0
- chore: Fix syntax highlighting in docs
- docs: Add documentation for value-matching methods
- fix: Address preview_domain() issue when there are no sample values (#92)


0.3.0 (2024-08-22)
------------------

Below is a detailed list of important changes included in this release.

New Features and Improvements: 
- feat: Consistently use np.nan for missing values
- feat: Add view_value_matches() method
- feat: Add new schema matcher based on value matching
- feat: Support passing method arguments to top_matches()
- feat: Support euclidean distance in CLTopkColumnMatcher

Documentation:
- docs: Add documentation for schema mapping methods
- docs: Add example for view_value_matches() method


0.2.0 (2024-07-26)
------------------

We are pleased to announce the release of bdikit version 0.2.0.
This version includes several new features, including a new API,
a new column-matching model based on contrastive learning,
several new matching algorithm implementations, new visualizations
to help with column-matching, and a new website with our API
documentation at https://bdi-kit.readthedocs.io/. Additionally,
we added lots of improvements to our development infrastructure,
including several unit tests, automated tests in our CI (continuous
integration) server, and automated release publication to PyPI.

Below is a detailed list of important changes included in this release.

New Features and Improvements: 
- feat: Add automatic model download (#34)
- feat: Adding AutoFuzzyJoin Algorithm for value mapping
- test: Add tests to TFIDF and EditDistance-based value matching
- feat: Format output text for unmatched values
- feat: Adding TwoPhase ColumnMatch algorithm based on CTLearning
- feat: Change column matching API to be stateless
- feat(api): Add new function bdi.match_columns()
- feat(api): Add new function bdi.top_matches()
- feat(api): Added bdi.materialize_mapping() and basic value mappers
- feat(api): add match_values() and preview_value_mappings()
- test(api): Add end-to-end API integration test
- feat(api): Add bdi.preview_domains()
- feat(api): Support an object as a method in match_columns()
- feat(api): Make API inputs more compatible
- feat: Change default algorithm for column mapping
- feat: Make EditAlgorithm to return values between 0 and 1
- feat: Release new constrastive learning model 'bdi-cl-v0.2' and make it default
- feat: Create a mapper from the output of preview_value_mappings
- feat: fit schema matching visualization tool to the new API
- feat: Update preview_domain() to include value names and descriptions (#68)
- feat: Allow configuring device (gpu/cpu) via env vars
- feat: Allow sending arguments for the matching algorithms (#75)
- feat: Adding embedding cache for gdc case (#76)

Infrastructure:
- Setup CI infrastructure for automated tests
- Add Python 3.11 to CI build script
- Setup GH action to check code format using black formatter
- Automatic code formatting using black
- Add 'format' and 'lint' targets to the Makefile
- infra: Add linter (pyright) config file
- feat: Setup PyPI release using GitHub Actions
- chore: Remove modules, transformation, data_ingestion, mapping_recommendation
- chore: Remove scope_reducing module
- chore: Remove empty module directories

Bug Fixes:
- Fix setup.py to read version correctly
- Temporarily use scipy<1.13 to avoid import error of triu package
- Temporarily use matplotlib<3.9 to avoid PolyFuzz error
- Fix conflicts and relax dependency versions
- TFIDFAlgorithm doesn't work for short values (#57)
- do not allow panel version 1.4.3 (fixes issue #40)
- Support repeated source columns in match_values()

Visualizations:
- Add accept and reject button and interaction (#39)
- Remove plot_reduce_scope from APIManager's reduce_scope
- Add ScopeReducerExplorer to column_and_value_mapping notebook
- Remove tabulate
- Use display() to show the visualization

Refactoring:
- Refactoring of column mappers to remove code duplication
- Moving model_name,top_k to construtor parameters for better class reuse
- Extract TopkColumnMatcher from ContrastiveLearningAPI 
- Rename match_columns() to match_schema() (#66)
- Renamed value matcher classes to use suffix 'ValueMatcher' 
- Merge preview_value_mappings() and match_values() 
- Make match_values() return a list of DataFrames
- Reorder API functions
- Rename update_mappings() to merge_mappings()
- Rename variables for better API naming consistency

Documentation:
- Add readthedocs configurations
- Update installation instructions
- Add API Reference documentation page for bdikit module functions
- Fix API documentation
- Updated README.md with code formatting instructions
- Rename notebook to doc_gdc_harmonization.ipynb
- Fix heading levels in getting-started.ipynb
- Fix typos and minor improvement in getting-started.ipynb
- Reorganize examples directory for documentation (#71)
- Minor formatting changes
- Improve API documentation and fix typos
- Improve documentation and minor typos
- Clarify description of MappingSpecLike


0.1.0 (2024-05-21)
-------------------

* Add a scope reducer for GDC.
* Add initial column mapping algorithms.
* Add initial value mapping algorithms.

# Change Log

## 0.10.0.dev0 (yyyy-mm-dd)


## 0.9.0 (2025-11-17)

We are pleased to announce the release of **BDI-Kit version 0.9.0**.  
This version introduces a new **Streamlit-based chatbot interface**, making it easier to interact with BDI-Kit through an intuitive, conversational UI.

Below is the main change included in this release:

### New Feature

- **Streamlit Chatbot Interface:** Added an interactive Streamlit-based chatbot for using BDI-Kit through a lightweight, browser-accessible interface. ([#138](https://github.com/VIDA-NYU/bdi-kit/pull/138))



## 0.8.0 (2025-10-20)

We are pleased to announce the release of **BDI-Kit version 0.8.0**.  
This version expands the toolkit’s AI integration capabilities with an enhanced **MCP server**, greater **flexibility for LLM-based primitives**, and multiple improvements in performance, compatibility, and documentation.

Below is a list of the main changes included in this release:


### New Feature

- **MCP Integration Entry Point:** Added a new entry point to simplify MCP integration and improve interoperability with AI assistants and external tools. ([#134](https://github.com/VIDA-NYU/bdi-kit/pull/134))  
- **Abstracted LLM Primitives:** Introduced an abstraction layer for LLM-based primitives, allowing users to specify their preferred LLM and pass additional arguments dynamically. ([#129](https://github.com/VIDA-NYU/bdi-kit/pull/129))  
- **LLM Integration with Magneto:** Enabled passing an LLM instance and extra arguments directly to the Magneto matcher. ([#130](https://github.com/VIDA-NYU/bdi-kit/pull/130))  


### Enhancement

- **Extended MCP Server:** Expanded the built-in MCP server with additional tools and utilities to enhance agent-based data harmonization workflows. ([#131](https://github.com/VIDA-NYU/bdi-kit/pull/131))  
- **Method Arguments Exposure:** Exposed `method_args` in the MCP server and enabled dynamic reading of the selected LLM for better configurability.  
- **BaseStandard Refactor:** Renamed internal `BaseStandard` functions for improved consistency across primitives. ([#128](https://github.com/VIDA-NYU/bdi-kit/pull/128))  
- **Deprecation Cleanup:** Removed deprecated methods and matchers, including `top_matches()`, `top_value_matches()`, `ct_learning`, and `fasttext`. Also removed related modules, tests, and docs. ([#125](https://github.com/VIDA-NYU/bdi-kit/pull/125))  


### Fix

- **PolyFuzz Match Handling:** Ensured PolyFuzz-based matchers always return valid value matches, even when no matches are found.  
- **Gensim Import Fix:** Added constraints for the Gensim library and corrected subpackage imports for stability. ([#137](https://github.com/VIDA-NYU/bdi-kit/pull/137))  
- **SciPy and Jupyter Compatibility:** Removed unnecessary SciPy version constraints and fixed Jupyter imports for smoother environment setup. ([#136](https://github.com/VIDA-NYU/bdi-kit/pull/136))  
- **Transformers Constraint:** Added a compatibility constraint to the `transformers` library to prevent dependency conflicts. ([#132](https://github.com/VIDA-NYU/bdi-kit/pull/132))  


### Documentation

- **Claude Integration Example:** Added a new example demonstrating how to integrate BDI-Kit with the Claude desktop interface. ([#127](https://github.com/VIDA-NYU/bdi-kit/pull/127))  
- **Contributing Guide:** Moved and expanded contributing instructions into the documentation site for easier access. ([#126](https://github.com/VIDA-NYU/bdi-kit/pull/126))  
- **General Fixes:** Improved clarity and structure in multiple documentation sections, including setup and usage examples. ([#135](https://github.com/VIDA-NYU/bdi-kit/pull/135))  
- **Dataset Additions:** Added the Clark dataset to support additional examples and testing scenarios.  



## 0.7.0 (2025-07-14)

We are pleased to announce the release of bdikit version 0.7.0.
This version brings powerful new capabilities such as the addition of a built-in MCP server for integration with AI agents and new primitives for value matching.

Below is a list of the main changes included in this release:


### New Feature

- **MCP Server:** Introduced an integrated MCP server to enable interaction with BDI-Kit via AI assistants and agent frameworks. ([#122](https://github.com/VIDA-NYU/bdi-kit/pull/122))
- **Evaluation of Schema and Value Matches:** Added methods to evaluate schema and value matches. ([#118](https://github.com/VIDA-NYU/bdi-kit/pull/118))
- **Contextual Matching Support:** Enabled users to attach contextual information to source or target datasets to improve matching quality. ([#117](https://github.com/VIDA-NYU/bdi-kit/pull/117))
- **Caching for Schema Matching:** Implemented a caching mechanism to avoid recomputing expensive match operations. ([#119](https://github.com/VIDA-NYU/bdi-kit/pull/119))
- **Value Matching Caching:** Added caching support for value matching functions to improve performance and reduce redundant computations. Also enhanced the caching mechanism for schema matching. ([#124](https://github.com/VIDA-NYU/bdi-kit/pull/124))
- **Numeric Mapping Support:** Introduced a numeric transformer primitive to handle numeric conversions during value matching. ([#112](https://github.com/VIDA-NYU/bdi-kit/pull/112))


### Enhancement

- **Match Filling Across Matchers:** Improved schema matcher consistency by filling in missing matches across methods. ([#115](https://github.com/VIDA-NYU/bdi-kit/pull/115))
- **MaxValSim Compatibility:** Updated the `max_val_sim` method to support newer schema and value matcher APIs.
- **Sorting and Completion of Matches:** Matches are now sorted to ensure uniformity across outputs.
- **Magneto as Default Matcher:** Set Magneto as the default schema matcher.
- **Valentine Matching Refinement:** Ensured strict one-to-one matching in Valentine-based matchers.


### API Change

- **LLM Method Renaming:** Renamed LLM-based methods for consistency and improved clarity. ([#121](https://github.com/VIDA-NYU/bdi-kit/pull/121))
- **Unification of Value Matching Output:** Standardized outputs from value matching methods to align with schema matching formats. ([#116](https://github.com/VIDA-NYU/bdi-kit/pull/116)) By default, the `match_values()` and `rank_value_matches()` functions now return a single DataFrame instead of a list of DataFrames. Note: This change is backward-incompatible.
- **Terminology Update:** Renamed `'columns'` to `'attributes'` in the outputs of various methods to ensure consistent terminology across the toolkit. ([#123](https://github.com/VIDA-NYU/bdi-kit/pull/123))


### Fix

- **NaN Handling in Value Matching:** Skipped NaN values in target attributes to avoid runtime errors.
- **Numeric String Handling:** Improved parsing of numbers represented as strings in matching pipelines.
- **GitHub Actions Compatibility:** Updated GitHub Action dependencies to maintain compatibility with Python signing workflows.


### Documentation

- **Table-to-Table Harmonization:** Added a comprehensive example demonstrating how to harmonize entire tables. ([#120](https://github.com/VIDA-NYU/bdi-kit/pull/120))
- **Quick-Start Guide:** Created a quick-start example to help new users begin using BDI-Kit more easily. ([#111](https://github.com/VIDA-NYU/bdi-kit/pull/111))
- **Numeric Mapper Documentation:** Documentation of the numeric transformer primitive for numeric data integration.
- **Versioned UI and Docs:** Enhanced the documentation site with version selectors and links. ([#110](https://github.com/VIDA-NYU/bdi-kit/pull/110))



## 0.6.0 (2025-04-15)

We are pleased to announce the release of bdikit version 0.6.0.
This version introduces exciting new features, such as similarity scores in match_schema() and improves support for Synapse integration.

Below is a list of the main changes included in this release:


### New Feature
- **Similarity Scores in Schema Matching**: Introduced similarity scores in `match_schema()` for enhanced matching accuracy. (#105)
- **Synapse Support**: Added support for Synapse integration. (#98)
- **Matcher Factory Tests**: Created tests for the matcher factory to improve test coverage.

### Enhancement
- **Reorganized Matchers**: Clearly differentiated between one-to-one and top-k matching methods. (#101)

### API Change
- **Refactored Matching Functions**: Updated and renamed functions with deprecation warnings to streamline the API. (#108)
- **`ct_learning` Method**: Deprecated the `ct_learning` method as part of ongoing improvements. (#96)
- **Deprecate functions**: Deprecated the functions `top_matches()` and `top_value_matches`. (#108)
- **Parameter Ordering in Top Matches**: Reordered parameters in `top_matches()` for consistency. (#99)

### Fix
- **Return All Original Source Values**: Addressed an issue where not all original source values were being returned.
- **Magneto Top-1 Bipartite Issue**: Resolved an issue with top-1 matching when using the Bipartite method.
- **PyTorch Compatibility**: Fixed an error when loading models with PyTorch version >=2.6.

### Documentation
- **Example Updates**: Revised examples to align with the current version.
- **Versioned Links**: Created links in the examples pointing to the current version. (#107)


0.5.0 (2025-01-17)
------------------

We are pleased to announce the release of bdikit version 0.5.0.
This version includes some new features, such as [Magneto](https://arxiv.org/pdf/2412.08194)  
as a schema matching method, and streamlines the integration of new algorithms and standards.

Below is a list of the main changes included in this release:

- feat: Restructure packages to add new algorithms easier (#87)
- feat: Streamline the addition of new standards (#88)
- feat: Add Magneto (#94)
- chore: Update the download and upload artifact in Github actions (#95)
- chore: Add an extra dependency to enable `flair` to load embeddings
- test: Add tests for more value matching methods


0.4.0 (2024-11-26)
------------------

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

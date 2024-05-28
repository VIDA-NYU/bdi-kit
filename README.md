[![PyPI version](https://badge.fury.io/py/bdi-kit.svg)](https://pypi.org/project/bdi-kit)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# bdi-kit 
This project aims to assist users in performing data integration on biomedical data. It provides tools to streamline the process of integrating disparate biomedical datasets.


## Installation

You can install the latest stable version of this library from [PyPI](https://pypi.org/project/bdi-kit/):

```
pip install bdi-kit
```

To install the latest development version:

```
pip install git+https://github.com/VIDA-NYU/bdi-kit@devel
```

Download the pre-trained model for mapping recommendations from [here](https://drive.google.com/file/d/1YdCTd-kUMjDJaltQwXN4X9ezTCsfjyft/view) (we show how to use it in our examples).


## Documentation
See our examples [here](https://github.com/VIDA-NYU/bdi-kit/tree/devel/examples).

## Contributing
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
black ./bdikit/
```

## Folder Structure

- **/data_ingestion**:
  - Contains scripts and tools for ingesting data into the system.

- **/mapping_algorithms**:
  - Algorithms and utilities for schema and value mapping.

- **/mapping_recommendation**:
  - Interacts with mapping_algorithms and users to suggest mappings for data integration. The output is a mapping plan.

- **/transformation**:
  - Transforms the data given a mapping plan.

- **/visualization**:
  - Visualizations to aid in the mapping recommendation process.

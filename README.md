[![PyPI version](https://badge.fury.io/py/bdi-kit.svg)](https://pypi.org/project/bdi-kit)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Documentation Status](https://readthedocs.org/projects/bdi-kit/badge/?version=latest)](https://bdi-kit.readthedocs.io/en/latest/?badge=latest)
[![Tests](https://github.com/VIDA-NYU/bdi-kit/actions/workflows/build.yml/badge.svg)](https://github.com/VIDA-NYU/bdi-kit/actions/workflows/build.yml)
[![Lint](https://github.com/VIDA-NYU/bdi-kit/actions/workflows/lint.yml/badge.svg)](https://github.com/VIDA-NYU/bdi-kit/actions/workflows/lint.yml)


# bdi-kit 

The `bdi-kit` is a library that assist users in performing data harmonization. It provides state-of-the-art tools to streamline the process of integrating and transforming disparate datasets (with a focus on biomedical data), and includes APIs for performing tasks such as:
- Schema matching
- Value matching
- Data transformation to a target schema/standard

**Warning:** `bdi-kit` is currently in *alpha* stage and under heavy development. Expect APIs to change.

## Documentation

Documentation is available at [https://bdi-kit.readthedocs.io/](https://bdi-kit.readthedocs.io/).


## Installation

You can install the latest stable version of this library from [PyPI](https://pypi.org/project/bdi-kit/):

```
pip install bdi-kit
```

To install the latest development version:

```
pip install git+https://github.com/VIDA-NYU/bdi-kit@devel
```


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
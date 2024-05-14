# Biomedical Data Integration 

## Overview
This project aims to assist users in performing data integration on biomedical data. It provides tools to streamline the process of integrating disparate biomedical datasets.

## Installation
This package works with Python 3.8+ in Linux, Mac, and Windows.

To install this library and the required dependencies, run:

```
1. git clone https://github.com/VIDA-NYU/askem-arpa-h-project.git
2. cd askem-arpa-h-project
3. pip install -e .
```

Download the pre-trained model for mapping recommendations from [here](https://drive.google.com/file/d/1YdCTd-kUMjDJaltQwXN4X9ezTCsfjyft/view).


[Coming soon] You can install the latest stable version of this library from [PyPI](#):

```
pip install bdi
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

# GDC Dictionary

## Introduction
The GDC Data Dictionary is a data schema provided by the GDC. In our case we are using this as a matcher guideline and formatter.
  

### Installation
```
pip install -r requirements.txt
```

### Usage
Load GDC schema as dictionary.
```
load_gdc_schema() -> dict
```
  
Match column name and return GDC metadata.
```
fetch_properties(column_name: str) -> dict
```
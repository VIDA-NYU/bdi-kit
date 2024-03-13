# GDC Dictionary

[toc]

## Introduction
The GDC Data Dictionary is a data schema provided by the GDC. In our case we are using this as a matcher guideline and formatter.
  

### Installation
```
pip install -r requirements.txt
```

### Example
[_Jupyter notebook_](https://github.com/VIDA-NYU/askem-arpa-h-project/blob/main/gdc/gdc.ipynb)

### Usage
#### Import
```from gdc_api import GDCSchema```

---

#### Create an instance of GDCSchema
```schema = GDCSchema('YOUR_COLUMN_OR_KEYWORD')``` or ```schema = GDCSchema()``` or ```schema = GDCSchema('YOUR_COLUMN_OR_KEYWORD', subschema='SUBSCHEMA')```

---

#### *Parse name-candidate schema matching from DataFrame
```schema.parse_df(dataset)```

---

#### Get GDC columns matches
```schema.get_gdc_candidates()```

---

#### Get GDC properties by candidate path
```schema.get_properties_by_gdc_candidate('CATEGORY::COL_NAME')```

---

#### Get GDC column type and values
```col_type = schema.get_gdc_col_type()```  
```col_type = schema.get_gdc_col_values()```  

---

#### Get GDC column descriptions
```description = schema.get_gdc_col_description()```

---

#### Set column name or keyword (if you did not set it upon create)
```schema.set_column_name('YOUR_COLUMN_OR_KEYWORD')```
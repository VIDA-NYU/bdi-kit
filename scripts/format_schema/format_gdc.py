import json
import pandas as pd
from os.path import join, dirname


RAW_GDC_PATH = join(dirname(__file__), "./gdc_raw_schema.json")
SYNTHETIC_GDC_PATH = join(dirname(__file__), "./gdc_synthetic.csv")
FORMATTED_GDC_PATH = join(dirname(__file__), "../../bdikit/resource/gdc_schema.json")

metadata = {}


with open(RAW_GDC_PATH) as json_file:
    gdc_schema = json.load(json_file)

gdc_stynthetic_data = pd.read_csv(SYNTHETIC_GDC_PATH)

for attrib_data in gdc_schema.values():
    for attrib_name, attrib_properties in attrib_data["properties"].items():
        metadata[attrib_name] = {}
        attrib_description = attrib_properties.get("description", "")
        metadata[attrib_name]["column_description"] = attrib_description

        value_names = attrib_properties.get("enum", [])

        descriptions = attrib_properties.get("enumDef", {})
        value_descriptions = []
        for value_name in value_names:
            description = ""
            if value_name in descriptions:
                description = descriptions[value_name].get("description", "")
            value_descriptions.append(description)
        
        metadata[attrib_name]["value_data"] = dict(zip(value_names, value_descriptions))

        # Store synthetic data for each attribute
        if attrib_name in gdc_stynthetic_data.columns:
            synthetic_values = gdc_stynthetic_data[attrib_name].unique()
        else:
            synthetic_values = []

        metadata[attrib_name]["synthetic_value_data"] =  {
            str(value_name) : ""
            for value_name in synthetic_values
        }
        

with open(FORMATTED_GDC_PATH, "w") as f:
    json.dump(metadata, f, indent=4)

print("GDC schema formatted successfully.")

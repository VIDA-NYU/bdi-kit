import json
from os.path import join, dirname


RAW_SYNAPSE_PATH = join(dirname(__file__), "./synapse_raw_schema.json")
FORMATTED_SYNAPSE_PATH = join(dirname(__file__), "../../bdikit/resource/synapse_schema.json")

data = {'entity':{}, 'subschema':{}}


with open(RAW_SYNAPSE_PATH) as json_file:
    sbn_schema = json.load(json_file)

for entry in sbn_schema["@graph"]:
    entry_name = entry["rdfs:label"]
    entry_description = entry.get("rdfs:comment", "")
    entry_parents = entry.get("rdfs:subClassOf", [{}])

    if entry_parents[0]['@id'] == "bts:Thing":

        if entry_name.endswith("Enum"):
            continue
        values = entry.get("schema:rangeIncludes", [])
        new_entry = {
            "column_description": entry_description,
            "value_data": {x["@id"].replace("bts:", ""): "" for x in values}
        }
        data['entity'][entry_name] = new_entry

    if entry_name.endswith("Template") and "sms:requiresDependency" in entry:
        dependencies = [x["@id"].replace("bts:", "") for x in entry["sms:requiresDependency"]]
        data['subschema'][entry_name] = dependencies

with open(FORMATTED_SYNAPSE_PATH, "w") as f:
    json.dump(data, f, indent=4)

print("Synapse schema formatted successfully.")

import pandas as pd
from openai import OpenAI
import json

def generate_prompt(column_name, column_values, column_type):
    if column_type == "Not specified":
        prompt = f"""Given a table with the header '{column_name}' and its values {column_values}, 
                use your biomedical expertise to identify two alternative names for this column as found in other datasets. 
                Ensure these names follow common database formatting conventions like underscores and abbreviations. 
                Also, provide possible synonyms or alternative forms for the values that are technically correct. Output in the standard json format:
                {{
                    "original_column": {{
                        "name": "{column_name}",
                        "values": "{column_values}",
                        "type": "{column_type}"
                    }},
                    "matches": [
                        {{
                            "name": "alternative_name_1",
                            "values": ["variant1", "variant2", "variant3"]
                        }},
                        {{
                            "name": "alternative_name_2",
                            "values": ["variant4", "variant5", "variant6"]
                        }}
                    ]
                }}
                Do not include any other information in your response.
                """
    else:
        prompt = f""" Given a table with the header '{column_name}' which has {column_type} data type,
                use your biomedical expertise to identify two alternative names for this column as found in other datasets. 
                Ensure these names follow common database formatting conventions like underscores and abbreviations. Output in the standard json format:
                {{
                    "original_column": {{
                        "name": "{column_name}",
                        "values": "{column_values}",
                        "type": "{column_type}"
                    }},
                    "matches": [
                        {{
                            "name": "alternative_name_1"
                        }},
                        {{
                            "name": "alternative_name_2"
                        }}
                    ]
                }}
                Do not include any other information in your response.
                """
    return prompt

OPENAI_API_KEY = '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' #TODO: Replace with your own OpenAI API key
client = OpenAI(api_key=OPENAI_API_KEY) 
gdc_schema = pd.read_csv('gdc_schema_in_gt.csv')

results = []

for index, row in gdc_schema.iterrows():
    print("Generating for", row['column name'])
    prompt = generate_prompt(row['column name'], row['column values'], row['column type'])
    response = client.chat.completions.create(model="gpt-4-turbo-preview",
                messages=[
                {
                    "role": "system", 
                    "content": "You are an agent specializing in schema matching for cancer research."},
                {
                    "role": "user", 
                    "content": prompt},
                ],
                temperature=0.3)
    content = response.choices[0].message.content
    content = content.strip('`json \n')
    print(content)
    results.append(json.loads(content))

with open('synthetic_matches.json', 'w') as f:
    json.dump(results, f, indent=4)

print("Done.")

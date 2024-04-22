import pandas as pd
from openai import OpenAI
import json
import random

def generate_prompt(column_name, column_values):
    prompt = f"""Given a table with the header '{column_name}' and its values {column_values}, 
            use your biomedical expertise to identify one alternative name for this column as found in other datasets. 
            Ensure this name follows common database formatting conventions like underscores and abbreviations. 
            Also, provide distinct possible synonyms or alternative forms for the values that are technically correct. Output in format:
            "alternative_name, value1, value2, value3, ..."
            Do not include any other information or use quotes in your response.
            """
    return prompt

OPENAI_API_KEY = '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' #TODO: Replace with your own OpenAI API key
client = OpenAI(api_key=OPENAI_API_KEY) 

gdc_schema = pd.read_csv('data/gdc_table.csv')

results = pd.DataFrame()
matches = []

count = 0
for column in gdc_schema.columns:
    column_name = column
    column_list_of_values = gdc_schema[column].tolist()
    column_list_of_values = [str(value) for value in column_list_of_values]
    column_values = ', '.join(column_list_of_values)
    prompt = generate_prompt(column_name, column_values)
    
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
    # split the response into the alternative name and values using the first comma
    alternative_name, values = content.split(', ', 1)
    print(alternative_name)
    list_of_values = values.split(', ')
    print(list_of_values)
    if len(list_of_values) > 15:
        list_of_values = list_of_values[:15]
    else:
        while len(list_of_values) < 15:
            list_of_values.append(list_of_values[random.randint(0, len(list_of_values) - 1)])
    
    results[alternative_name] = list_of_values
    matches.append({'l_column_id': column, 'r_column_id': alternative_name})
    count += 1
    
    if count % 10 == 0:
        print(f"Processed {count} columns")
        results.to_csv('gdc_table_synthetic.csv', index=False)

results.to_csv('gdc_table_synthetic.csv', index=False)

matches_df = pd.DataFrame(matches)
matches_df.to_csv('train.csv', index=False)

print("Done.")

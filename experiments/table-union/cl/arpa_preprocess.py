import pandas as pd
from openai import OpenAI
import random
import json
import csv

def extract_properties():
    with open('../askem-arpa-h-project/data/use_case1/gdc_schema.json', 'r') as f:
        schema_data = json.load(f)
    output_filename = '../askem-arpa-h-project/table-union/gdc_schema_description.csv'
    
    properties_info = []
    for schema_key, schema_value in schema_data.items():
        if 'properties' in schema_value:
            for prop, attributes in schema_value['properties'].items():
                # check if prop already exists in properties_info
                if not any(d['column name'] == prop for d in properties_info): 
                    if 'enum' not in attributes:
                        column_type = 'Not specified'
                        max_value = 'Not specified'
                        min_value = 'Not specified'
                        description = 'Not specified'
                        
                        column_type = attributes.get('type')
                        description = attributes.get('description', 'Not specified')
                        
                        if 'oneOf' in attributes:
                            for option in attributes['oneOf']:
                                if 'maximum' in option:
                                    max_value = option['maximum']
                                if 'minimum' in option:
                                    min_value = option['minimum']             
                        
                        # if description != 'Not specified':
                        properties_info.append({
                            'column name': prop,
                            'column type': column_type,
                            'description': description,
                            'maximum': max_value,
                            'minimum': min_value
                        })

    with open(output_filename, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['column name', 'column type', 'description', 'maximum', 'minimum'])
        writer.writeheader()
        writer.writerows(properties_info)


def prompt_gpt(column_name, column_type, description, maximum, minimum):
    if column_type != "Not specified":
        c_type = f" and type '{column_type}'"
    else:
        c_type = ""
    
    if description != "Not specified":
        c_description = f"The description of the column is '{description}'"
    else:
        c_description = ""
    
    if maximum != "Not specified" and minimum != "Not specified":
        c_range = f" with a range of {minimum} to {maximum}"
    elif maximum != "Not specified":
        c_range = f" with a maximum value of {maximum}"
    elif minimum != "Not specified":
        c_range = f" with a minimum value of {minimum}"
    else:
        c_range = ""
    
    OPENAI_API_KEY = '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' #TODO: Replace with your own OpenAI API key
    client = OpenAI(api_key=OPENAI_API_KEY) 
    
    prompt = f"""Generate a list of 15 distinct values for a column with the header '{column_name}'{c_type}{c_range}, \
drawing on your biomedical expertise and domain knowledge. {c_description} \
Separate each value with a comma and do not include any additional information.
"""
    # print(prompt)
    response = client.chat.completions.create(model="gpt-4-turbo-preview",
                messages=[
                {
                    "role": "system", 
                    "content": "You are an agent specializing in generating column values for cancer research."},
                {
                    "role": "user", 
                    "content": prompt},
                ],
                temperature=0.3)
    content = response.choices[0].message.content
    return content

def convert_to_table():
    data = pd.read_csv('../data/gdc_schema_extracted.csv')
    data_description = pd.read_csv('../data/gdc_schema_description.csv')
    result_table = pd.read_csv('../data/gdc_table.csv')

    # for index, row in data.iterrows():
    # for last 5 rows
    for index, row in data.iloc[-5:].iterrows():
        if row['column type'] == 'boolean':
            list_of_values = ['True', 'False']
            while len(list_of_values) < 15:
                list_of_values.append(list_of_values[random.randint(0, 1)])
            print(f'{row["column name"]}: {list_of_values}')
        elif row['column values'] != "Not specified":
            list_of_values = row['column values'].split(', ')
            if len(list_of_values) > 15:
                list_of_values = list_of_values[:15]
            else:
                while len(list_of_values) < 15:
                    list_of_values.append(list_of_values[random.randint(0, len(list_of_values) - 1)])
            print(f'{row["column name"]}: {list_of_values}')
        else:
            row_description = data_description.loc[data_description['column name'] == row['column name']]
            if row_description.empty:
                continue
            else:
                row_description = row_description.fillna('Not specified')
                column_name = row_description['column name'].values[0]
                column_type = row_description['column type'].values[0]
                description = row_description['description'].values[0]
                maximum = row_description['maximum'].values[0]
                minimum = row_description['minimum'].values[0]
                content = prompt_gpt(column_name, column_type, description, maximum, minimum)
                print(f'{row["column name"]}: {content}')
                list_of_values = content.split(', ')
        
        if len(list_of_values) != 15:
            print(f'Column {row["column name"]} does not have 15 values')
        else:
            result_table[row['column name']] = list_of_values
        
        if index % 10 == 0:
            print(f'Processed {index} columns')
            result_table.to_csv('../data/gdc_table.csv', index=False)
    
    result_table.to_csv('../data/gdc_table.csv', index=False)
    
def generate_random_tables():
    table = pd.read_csv('gdc_table.csv')
    index = 1
    while True:
        if index < 10:
            filename = f'gdc_train/gdc_table_0{index}.csv'
        else:
            filename = f'gdc_train/gdc_table_{index}.csv'
        if table.shape[1] <= 20:
            table.to_csv(filename, index=False)
            break
        num_columns = random.randint(5, 20)
        columns = table.sample(num_columns, axis=1)
        columns.to_csv(filename, index=False)
        index += 1
        table = table.drop(columns.columns, axis=1)
        
    
if __name__ == '__main__':
    # extract_properties()
    convert_to_table()
    # generate_random_tables()
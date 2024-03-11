import pandas as pd
import os

current_directory = os.getcwd()



if 'data/table-matching-ground-truth' not in current_directory:
    current_directory = os.path.join(current_directory, "data/table-matching-ground-truth")


file_path = os.path.join(current_directory, "data_column_ground_truth.xlsx")
excel_data = pd.read_excel(file_path, sheet_name=None)

for sheet_name, sheet_data in excel_data.items():      
    
    study_name = sheet_name.split('-')[0].strip()
    print('Extracting the groundtruth for ', study_name)  
    columns = sheet_data[['original_paper_variable_names', 'GDC_format_variable_names']]
    df = pd.DataFrame(columns)
    df = df[df['GDC_format_variable_names'].notnull()]

    if not df.empty:
        file_path = os.path.join(current_directory, 'ground-truth')   
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        file_name = os.path.join(file_path, study_name + '.csv')
        print('Saving the ground-truth for ', study_name, ' in ', file_name)
        df.to_csv(file_name, index=False)
        
    
    


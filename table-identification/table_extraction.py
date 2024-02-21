import os
import pandas as pd

BASE_DIRECTORY = "data/datalake"
EXTRACTED_TABLES_DIRECTORY = "data/extracted-tables"
MIN_COLUMNS = 4
MIN_ROWS = 4

#TODO: Add more (smarter) checks to validate the table
def is_valid_table(df):
    if df.empty:
        return False
    if len(df.columns) < MIN_COLUMNS:
        return False
    if len(df) < MIN_ROWS:
        return False
    return True

def from_excel_to_csv(full_filename):
    # print(f'Processing {full_filename} ...')
    excel_data = pd.read_excel(full_filename, sheet_name=None)
    for sheet_name, sheet_data in excel_data.items():
        df = pd.DataFrame(sheet_data)
        if is_valid_table(df):
            csv_filename = f'{EXTRACTED_TABLES_DIRECTORY}/{os.path.basename(full_filename.replace('.xlsx',''))}_{sheet_name}.csv'
            csv_filename = csv_filename.replace(' ', '_')
            print(f'Writing {csv_filename} ...')
            df.to_csv(csv_filename, index=False)

            
        
def handle_files(directory = BASE_DIRECTORY):
    for filename in sorted(os.listdir(directory)):
        full_filename = os.path.join(directory, filename)
        #Excel files
        if os.path.isfile(full_filename) and filename.endswith(".xlsx"):
            from_excel_to_csv(full_filename)
        #TODO: Add support for other file types
            
def main():
    handle_files()

if __name__ == "__main__":
    main()


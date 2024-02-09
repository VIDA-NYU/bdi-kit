import os
import pandas as pd

directory = "raw-data"

output_directory = "csv-data"
os.makedirs(output_directory, exist_ok=True)

for filename in os.listdir(directory):
    if filename.endswith(".xlsx"):
        file_path = os.path.join(directory, filename)
        df = pd.read_excel(file_path)
        
        csv_file_path = os.path.join(output_directory, f"{os.path.splitext(filename)[0]}.csv")
        df.to_csv(csv_file_path, index=False)

        print(f"Converted {filename} to CSV.")

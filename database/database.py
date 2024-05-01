import pandas as pd
import os

class Database:
    """
    A class representing a database that stores dataframes.

    Attributes:
        dataframes (dict): A dictionary to store dataframes.

    Methods:
        load_data(df_name, file_path): Load data from a CSV file into a dataframe and store it in the database.
        load_data_from_folder(folder_path): Load data from all CSV files in a folder.
        get_dataframe(df_name): Retrieve a dataframe by its name.
        describe_database(): Print out the names, shape, columns, and head of all dataframes stored in the database.
    """
    
    def __init__(self):
        self.dataframes = {}

    def load_data(self, df_name, file_path):
        """
        Load data from CSV file into a dataframe and store it in the database.

        Args:
            df_name (str): Name to identify the data.
            file_path (str): Path to the CSV file to load.
        """
        if df_name in self.dataframes:
            raise ValueError(f"Dataframe associated with file name '{df_name}' already exists in the database.")
        
        df = pd.read_csv(file_path)
        self.dataframes[df_name] = df
    
    def load_data_from_folder(self, folder_path):
        """
        Function to load data from all CSV files in a folder using the Database class.

        Args:
            folder_path (str): Path to the folder containing CSV files.
        """    
        for file_name in os.listdir(folder_path):
            # print(file_name)
            if file_name.endswith(".csv"):
                file_path = os.path.join(folder_path, file_name)
                self.load_data(file_name, file_path)

    def get_dataframe(self, df_name):
        """
        Retrieve dataframe by df_name.

        Args:
            df_name (str): Name of the dataframe name.

        Returns:
            pd.DataFrame: Dataframe associated with the given file name.
        """
        return self.dataframes.get(df_name)
    
    def describe_database(self):
        """
        Print out the names of all dataframes stored in the database.
        """
        print("Database contains the following dataframes:")
        for df_name in self.dataframes:
            print(f"\t- {df_name}")
            print(f"\t\t- Shape: {self.dataframes[df_name].shape}")
            print(f"\t\t- Columns: {self.dataframes[df_name].columns}")
            # print(f"\t\t- Data types: {self.dataframes[df_name].dtypes}")
            # print(f"\t\t- Head: \n{self.dataframes[df_name].head()}")


# current_directory = os.getcwd()
# print("Current directory:", current_directory)
# folder_path = "data/extracted-tables/"
# db = Database()
# # filemame = "Cao_Clinical_data.csv"
# # file_path = os.path.join(folder_path, filemame)
# # db.load_data(filemame, file_path)
# db.load_data_from_folder(folder_path)
# db.describe_database()
# print(dir(db))
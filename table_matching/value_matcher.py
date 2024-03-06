import pandas as pd
import numpy as np

#TODO refactor
#TODO check other types
#TODO Semantic type annotation
def get_type(value):
    if type(value) == float and value is not np.nan:
        return float
    elif type(value) == int and value is not np.nan:
        return int
    value = str(value) #TODO check also dates, and other types
    if value.isnumeric():
        if "." in value:
            return float
        else:
            return int
    else:
        return str

def get_unique_data_types(column):
    return set(get_type(x) for x in column)


def detect_types(df: pd.DataFrame) -> dict:
    types = {}
    for col in df.columns:
        types[col] = get_unique_data_types(df[col])
    print(types)
    return types
    

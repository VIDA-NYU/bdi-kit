import pandas as pd


def load_table_matching_groundtruth(datapath):
    """
    Load a the groundtruth from a CSV file and return a list of tuples where each tuple represents a mapping from candidate to target.
    Args:
        datapath (str): The path to the CSV file.
    Returns:
        list: A list of tuples, where each tuple represents a mapping from candidate to target.
    """
    df = pd.read_csv(datapath)
    # Create a list of tuples from two columns of the dataframe
    # groundtruth = list(zip(df['candidate'], df['target']))
    groundtruth = list(zip(df['original_paper_variable_names'], df['GDC_format_variable_names']))
    return groundtruth

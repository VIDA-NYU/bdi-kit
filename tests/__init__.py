import os
import pandas as pd

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import utils.file_utils as ut

# Load the data for the tests
script_dir = os.path.dirname(__file__)

candidate_path = os.path.join(script_dir, 'test-data', 'Dou-data.csv')
target_path = os.path.join(script_dir, 'test-data', 'target.csv')
ground_truth_path = os.path.join(script_dir, 'test-data', 'Dou-groundtruth.csv')

candidate_df = pd.read_csv(candidate_path)
target_df = pd.read_csv(target_path)
ground_truth = ut.load_table_matching_groundtruth(ground_truth_path)


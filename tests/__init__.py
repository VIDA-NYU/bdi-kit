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


integration_path0 = os.path.join(script_dir, 'test-data','integration', 'integration0.csv')
integration_path1 = os.path.join(script_dir, 'test-data','integration', 'integration1.csv')
integration_path2 = os.path.join(script_dir, 'test-data','integration', 'integration2.csv')

integration0_df = pd.read_csv(integration_path0)
integration1_df = pd.read_csv(integration_path1)
integration2_df = pd.read_csv(integration_path2)


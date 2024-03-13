
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from tests import candidate_df, target_df, ground_truth
from table_matching.table_matcher import detect_matching_columns

# import unittest
#TODO use unitetest




detect_matching_columns(candidate_df, target_df, ground_truth)

import sys
import os
import glob
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from tests import integration0_df, integration1_df, integration2_df, script_dir
from table_integration.naive_integrator import naive_integration
from table_integration.fast_full_disjunction import FDAlgorithm

# import unittest
#TODO use unitetest


d = naive_integration([integration0_df, integration1_df, integration2_df])



cluster = 'integration'
integration_path = os.path.join(script_dir, 'test-data','integration')
filenames = sorted(glob.glob(integration_path + "/*.csv"))


fd_table, stats_df, debug_dict = FDAlgorithm(filenames, cluster)

print('\n')

print(d.head(15))
print('\n')
print(fd_table.head(15))
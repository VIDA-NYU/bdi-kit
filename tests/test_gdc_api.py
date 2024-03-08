import unittest
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from gdc.gdc_api import GDCSchema


gdc_schema = GDCSchema("days_to_birth")
# gdc_schema.get_properties_by_gdc_candidate('days_to_birth')
gdc_schema.get_properties_by_gdc_candidate(list(gdc_schema.candidates.keys()[0]))
# gdc_schema.get_gdc_col_type() # return "integer"
# gdc_schema.get_gdc_col_values() # return (-32872, 0)
# gdc_schema.get_gdc_col_description() # return "Number of days between ..."


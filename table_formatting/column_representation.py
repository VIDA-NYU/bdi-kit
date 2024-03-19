import os
import sys
import numpy as np
from enum import Enum

from typing import Optional, List, Callable, Any

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from gdc.gdc_api import GDCSchema

class DataType(Enum):
        STRING = 'string'
        FLOAT = 'float'
        INTEGER = 'integer'

class ColumnRepresentation:
    def __init__(self, column_name: str, column_type: str, null_values_representations: List, possible_column_values: Optional[List[str]] = None) -> None:
        self.column_name: str = column_name
        self.column_type: str = column_type
        self.null_values_representations: List = null_values_representations
        
        null_set = set(null_values_representations)
        if possible_column_values is not None:
            self.possible_column_values: List[str] = [v.lower() for v in possible_column_values if v.lower() not in null_set]
        else:
            self.possible_column_values: Optional[List[str]] = None
        
        self.check_constraints_list: List[Callable[[Any], bool]] = []

    def __str__(self) -> str:
        return f"ColumnRepresentation(column_name={self.column_name}, column_type={self.column_type}, null_values_representations={self.null_values_representations}, possible_column_values={self.possible_column_values})"

    def add_check_constraint(self, constraint: Callable[[Any], bool]) -> None:
        self.check_constraints_list.append(constraint)

    def check_constraints(self, value: Any) -> bool:
        return all(constraint(value) for constraint in self.check_constraints_list)


print('Loading GDC information (column domain specification) for columns in target (compiled data)...')
gdc_target_columns = []

# Categorical columns
cat_null_values = ['n/a', 'na','nan','null', 'lost to follow-up', 'not available', 'unable to obtain',
                        'unknown', 'not reported', 'not allowed to collect',
                        'unspecified', 'not specified']

def add_categorical_col(subschema, cols):
    for col in cols:
        colname = subschema + '::' + col
        sc = GDCSchema(col, subschema)
        values = set(sc.get_properties_by_gdc_candidate(colname)['enum']) 
        coltype = DataType.STRING
        col_representation = ColumnRepresentation(colname, coltype, cat_null_values, values)
        gdc_target_columns.append(col_representation)
        print(col_representation.column_name,'with' ,len(col_representation.possible_column_values), 'possible values (excluding null representations).')

subschema = 'demographic'
cols = [ 'vital_status','ethnicity', 'gender', 'race']
add_categorical_col(subschema, cols)

subschema = 'diagnosis'
cols = ['ajcc_pathologic_t','tumor_grade','primary_diagnosis','ajcc_pathologic_n','tumor_focality','tissue_or_organ_of_origin','ajcc_pathologic_stage','morphology']
add_categorical_col(subschema, cols)

subschema = 'sample'
cols = [ 'tumor_code']
add_categorical_col(subschema, cols)

# Numerical columns
numeric_null_values = [np.nan]
def within_range(min_value, max_value) -> Callable[[Any], bool]:
    def _within_range(value: Any) -> bool:
        return min_value <= value <= max_value
    return _within_range

subschema = 'clinical'
col = 'age_at_diagnosis'
colname = subschema + '::' + col
coltype = DataType.INTEGER
col_representation = ColumnRepresentation(colname, coltype, numeric_null_values)
col_representation.add_check_constraint(within_range(0, 140*365 )) # 140 years in days
# for v in [1, 500, -1, 10000000]:
#         print(v, col_representation.check_constraints(v))
print(col_representation.column_name,'with', col_representation.check_constraints_list, 'check constraints.')
gdc_target_columns.append(col_representation)


subschema = 'pathology_detail'
col =  'tumor_largest_dimension_diameter'
colname = subschema + '::' + col
coltype = DataType.FLOAT
col_representation = ColumnRepresentation(colname, coltype, numeric_null_values)
col_representation.add_check_constraint(within_range(0, 50 )) # 50cm centimeters
print(col_representation.column_name,'with', col_representation.check_constraints_list, 'check constraints.')
gdc_target_columns.append(col_representation)

from bdikit.mapping_algorithms.column_mapping.algorithms import (
    SimFloodAlgorithm,
    ComaAlgorithm,
    CupidAlgorithm,
    DistributionBasedAlgorithm,
    JaccardDistanceAlgorithm,
    GPTAlgorithm,
)
from enum import Enum


class MappingAlgorithm(Enum):
    SIMFLOOD = "SimFloodAlgorithm"
    COMA = "ComaAlgorithm"
    CUPID = "CupidAlgorithm"
    DISTRIBUTION_BASED = "DistributionBasedAlgorithm"
    JACCARD_DISTANCE = "JaccardDistanceAlgorithm"
    GPT = "GPTAlgorithm"


class ColumnMappingManager:
    def __init__(
        self, dataset, global_table, algorithm=MappingAlgorithm.SIMFLOOD.value
    ):
        self._dataset = (
            dataset  # TODO: move into database object (in data_ingestion folder)
        )
        self._global_table = (
            global_table  # TODO: move into database object (in data_ingestion folder)
        )
        self._reduced_scope = (
            None  # TODO: move into database object (in data_ingestion folder)
        )
        self.mapping_algorithm = algorithm

    @property
    def reduced_scope(self):
        return self._reduced_scope

    @reduced_scope.setter
    def reduced_scope(self, value):
        self._reduced_scope = value

    @property
    def dataset(self):
        return self._dataset

    @property
    def global_table(self):
        return self._global_table

    def map(self):
        if self.reduced_scope is None:
            mapping_algorithm_instance = eval(self.mapping_algorithm)(
                self.dataset, self.global_table
            )
            mappings = mapping_algorithm_instance.map()
            return mappings
        else:
            # For each reduction suggestion, we build a new dataset and global table and run the mapping algorithm
            mappings = {}
            for reduction in self.reduced_scope:
                dataset_column = reduction["Candidate column"]
                global_table_columns = [x[0] for x in reduction["Top k columns"]]

                if dataset_column in self.dataset.columns:
                    reduced_dataset = self.dataset[[dataset_column]]
                else:
                    continue

                common_cols = set(global_table_columns).intersection(
                    self.global_table.columns
                )
                reduced_global_table = self.global_table[list(common_cols)]

                mapping_algorithm_instance = eval(self.mapping_algorithm)(
                    reduced_dataset, reduced_global_table
                )
                partial_mappings = mapping_algorithm_instance.map()

                if len(partial_mappings.keys()) > 0:
                    candidate_col = next(iter(partial_mappings))
                    target_col = partial_mappings[candidate_col]
                    mappings[candidate_col] = target_col

            return mappings

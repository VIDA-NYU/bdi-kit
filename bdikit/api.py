from bdikit.data_ingestion.dataset_loader import load_dataframe
from bdikit.mapping_recommendation.scope_reducing_manager import ScopeReducingManager
from bdikit.mapping_recommendation.value_mapping_manager import ValueMappingManager
from bdikit.mapping_recommendation.column_mapping_manager import ColumnMappingManager
from bdikit.visualization.mappings import (
    plot_reduce_scope,
    plot_column_mappings,
    plot_value_mappings,
)
from bdikit.utils import get_gdc_data
from os.path import join, dirname
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable huggingface messages

GDC_DATA_PATH = join(dirname(__file__), "./resource/gdc_table.csv")


class APIManager:
    def __init__(
        self,
    ):
        """
        Create/instantiate an APIManager object.
        """
        # TODO: move into database object (in data_ingestion folder)
        self.dataset = None
        # TODO: move into database object (in data_ingestion folder)
        self.global_table = None
        # Assuming self.reduced_scope to be a list o dictionary (one dict for each column)TODO: move into database object (in data_ingestion folder)
        self.reduced_scope = None
        self.column_manager = None
        self.value_manager = None
        self.column_mappings = None  # TODO move this to a property in column_manager
        self.value_mappings = None  # TODO move this to a property in value_manager

    def load_global_table(self, global_table_path=None):
        """
        Load a global/target table.

        :param global_table_path: Path to the CSV dataset.
        """
        if global_table_path is None:
            self.global_table = load_dataframe(GDC_DATA_PATH)
        else:
            self.global_table = load_dataframe(global_table_path)
        return self.global_table

    def load_dataset(self, dataset_path):
        """
        Load the dataset.

        :param dataset_path: Path to the CSV dataset.
        """
        if self.global_table is None:
            self.load_global_table()
        self.dataset = load_dataframe(dataset_path)

        return self.dataset

    def reduce_scope(self):
        """
        Reduce the scope of the target domain. Supports only GDC format.
        """
        self.scope_manager = ScopeReducingManager(self.dataset, self.global_table)
        self.reduced_scope = self.scope_manager.reduce()
        plot_reduce_scope(self.reduced_scope, self.dataset)

    def map_columns(self, algorithm="SimFloodAlgorithm"):
        """
        Map columns.

        :param algorithm: The algorithm to perform the colum mappings. Supports: `SimFloodAlgorithm`,
            `ComaAlgorithm`, `CupidAlgorithm`, `DistributionBasedAlgorithm`, `JaccardDistanceAlgorithm`,
            and `GPTAlgorithm`.
        """
        self.column_manager = ColumnMappingManager(
            self.dataset, self.global_table, algorithm
        )
        self.column_manager.reduced_scope = self.reduced_scope
        self.column_mappings = self.column_manager.map()
        plot_column_mappings(self.column_mappings)

        return self.column_mappings

    def map_values(self, algorithm="EditAlgorithm"):
        """
        Map column values.

        :param algorithm: The algorithm to perform the value mappings. Supports: `EditAlgorithm`,
            `TFIDFAlgorithm`, `EmbeddingAlgorithm` and `LLMAlgorithm`.

        """
        self.global_table_all = get_gdc_data(self.column_mappings.values())
        self.value_manager = ValueMappingManager(
            self.dataset, self.column_mappings, self.global_table_all, algorithm
        )
        self.value_mappings = self.value_manager.map()
        plot_value_mappings(self.value_mappings)

        return self.value_mappings

    def update_reduced_scope(
        self, original_column, new_candidate_name, new_candidate_sim=1.0
    ):
        """
        Update the values returned by the scope reducer algorithm.

        :param original_column: Name of the original column.
        :param new_candidate_name: New name of the candidate column.
        :param new_candidate_sim: New similarity of the candidate column.
        """
        for index in range(len(self.reduced_scope)):
            if self.reduced_scope[index]["Candidate column"] == original_column:
                self.reduced_scope[index]["Top k columns"].append(
                    (new_candidate_name, new_candidate_sim)
                )
                print("Reduced scope updated!")
                plot_reduce_scope(self.reduced_scope)
                break

    def update_column_mappings(self, new_mappings):
        """
        Update the column mappings.

        :param new_mappings: List of tuples (original_column, correct_target_column).
        """
        for original_column, new_target_column in new_mappings:
            self.column_mappings[original_column] = new_target_column

        print("Column mapping updated!")
        plot_column_mappings(self.column_mappings)

    def update_value_mappings(
        self, original_column, original_value, new_target_value, new_similarity=1.0
    ):
        """
        Update the column value mappings.

        :param original_column: Name of the original column.
        :param original_value: Name of the original column value.
        :param new_target_value: New name of the target column value.
        :param new_similarity: New similarity of the original and target column values.
        """
        for index in range(len(self.value_mappings[original_column]["matches"])):
            if (
                self.value_mappings[original_column]["matches"][index][0]
                == original_value
            ):
                self.value_mappings[original_column]["matches"][index] = (
                    original_value,
                    new_target_value,
                    new_similarity,
                )
                print("Value mapping updated!")
                plot_value_mappings(self.value_mappings)
                break

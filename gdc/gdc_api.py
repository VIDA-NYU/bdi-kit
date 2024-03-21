import json
import logging
import os
from os.path import dirname, join

import jellyfish

logger = logging.getLogger(__name__)

PATH_TO_GDC_SCHEMA = join(dirname(__file__), "gdc_schema.json")


class GDCSchema:
    """
    GDCSchema class is used to get the GDC schema information based on the input column name.
    It provides methods to get the candidate column names, column type, column values, and column description.

    Example:
    ```
    gdc_schema = GDCSchema("days_to_birth")
    gdc_schema.get_properties_by_gdc_candidate(list(gdc_schema.candidates.keys()[0]))
    gdc_schema.get_gdc_col_type() # return "integer"
    gdc_schema.get_gdc_col_values() # return (-32872, 0)
    gdc_schema.get_gdc_col_description() # return "Number of days between ..."
    ```

    :param schema: dict, the GDC schema information
    :param column_name: str, the input column name
    :param properties: dict, the properties of candidate gdc column name
    :param candidates: dict, the candidate gdc column names and their similarity score
    """

    def __init__(self, column_name=None, subschema=None):
        self.schema = load_gdc_schema()
        self.properties = None
        
        self.subschema = None
        if subschema:
            if subschema not in self.schema.keys():
                logger.error("Invalid subschema, make sure your subschema is in schema.keys!")
            else:
                self.subschema = subschema
        
        if column_name is not None:
            self.set_column_name(column_name)
        else:
            self.column_name = None
            self.candidates = None

    
    # Static methods
    @staticmethod
    def compute_jaro_score(values, choices):
        """
        Compute the Jaro similarity score between the input values and the choices.
        For each value, it will find the maximum Jaro similarity score with the choices
        and then return the average score.

        :param values: list, the input values
        :param choices: list, the choices to compare with from GDC enums
        :return: score: float, the average Jaro similarity score
        """
        score = 0
        for value in values:
            score += max([jellyfish.jaro_similarity(value, choice) for choice in choices])
        return score / len(values)
    
    @staticmethod
    def compute_embedded_col_values_name_score(df_enums, candidate_enums):
        """
        Compute Jaro Score (see: compute_jaro_score) and column name similarity score.
        Output is sorted by Jaro Score and then by column name similarity score.

        :param df_enums: dict, the input column names and their unique values (should be lower cased)
        :param candidate_enums: dict, the candidate column names and their unique values (should be lower cased)
        :return: embedded_scores: dict, the embedded scores of column names and their candidate column names
        """
        embedded_scores = {}
        for col_name, values in df_enums.items():
            scores = {}
            for candidate, choices in candidate_enums.items():
                name_score = jellyfish.jaro_similarity(col_name, candidate)
                scores[candidate] = {
                    "jaro": GDCSchema.compute_jaro_score(values, choices),
                    "name": name_score,
                }
            scores = {k: v for k, v in sorted(scores.items(), key=lambda item: (item[1]["jaro"], item[1]["name"]), reverse=True)}
            # print(f"{col_name}: {list(scores.keys())[0]}")
            embedded_scores[col_name] = scores
        return embedded_scores
            
    def parse_df(self, df):
        matches = {}
        for col_name in df.columns:
            candidate = list(self.get_gdc_candidates(col_name).keys())
            if candidate:
                _ = self.get_properties_by_gdc_candidate(candidate[0]) # this will automatically set the properties
                matches[col_name] = {
                    "candidate": candidate[0],
                    "type": self.get_gdc_col_type(),
                    "values": self.get_gdc_col_values(),
                }
            else:
                matches[col_name] = {}
        return matches
    
    def extract_enums(self, subschemas=None):
        enums = {}
        for subschema, values in self.get_schema().items():
            if subschemas and subschema not in subschemas:
                continue
            for col_name, properties in values["properties"].items():
                if "enum" in properties:
                    if col_name not in enums:
                        enums[col_name] = []
                    enums[col_name].extend([value.lower() for value in properties["enum"]])
        for col_name, values in enums.items():
            enums[col_name] = list(set(values))
        
        return enums

    def _check_properties_valid(function):
        def magic(self):
            if self.properties is None:
                logger.error("Please run get_properties_by_gdc_candidate method first!")
                return KeyError(
                    "Please run get_properties_by_gdc_candidate method first!"
                )
            return function(self)

        return magic

    def get_gdc_candidates(self, column_name=None):
        """
        Get the candidates of GDC column names based on the similarity of the input column name.
        Need to run set_column_name first to make sure the column_name is set.

        :return: candidates: dict of candidate column names and their similarity score
        """
        if self.get_column_name() is None and column_name is None:
            logger.error("Please run set_column_name first!")
            return {}
        
        if not column_name:
            column_name = self.column_name
        candidates = {}
        
        if self.subschema:
            items = {self.subschema:self.get_schema()[self.subschema]}.items()
        else:
            items = self.get_schema().items()
            
        for parent, values in items:
            for key in values["properties"].keys():
                discription = ""
                if "description" in values["properties"][key]:
                    discription = values["properties"][key]["description"]
                elif "common" in values["properties"][key]:
                    discription = values["properties"][key]["common"]["description"]
                if column_name == key:
                    candidates[f"{parent}::{key}"] = {
                        "score": 1,
                        "description": discription,
                    }
                elif jellyfish.jaro_similarity(column_name, key) > 0.7:
                    candidates[f"{parent}::{key}"] = {
                        "score": jellyfish.jaro_similarity(column_name, key),
                        "description": discription,
                    }
        return dict(
            sorted(candidates.items(), key=lambda x: x[1]["score"], reverse=True)
        )

    def get_properties_by_gdc_candidate(self, gdc_colname):
        parent, colname = gdc_colname.split("::")
        if (
            parent in self.get_schema()
            and colname in self.get_schema()[parent]["properties"]
        ):
            self.set_properties(self.get_schema()[parent]["properties"][colname])
            return self.get_properties()
        return KeyError(
            "No such column name in GDC schema"
            "please check the valid gdc candidate column names!"
            f"Valid candidates are: {self.candidates.keys()}"
        )

    @_check_properties_valid
    def get_gdc_col_type(self):
        if "enum" in self.properties:
            return "enum"
        elif "type" in self.properties:
            return self.properties["type"]
        else:
            return None


    @_check_properties_valid
    def get_gdc_col_values(self):
        col_type = self.get_gdc_col_type()
        if col_type == "enum":
            return self.properties["enum"]
        elif col_type == "number" or col_type == "integer" or col_type == "float":
            return (
                self.properties["minimum"]
                if "minimum" in self.properties
                else -float("inf"),
                self.properties["maximum"]
                if "maximum" in self.properties
                else float("inf"),
            )
        elif col_type == "boolean":
            return [True, False]
        else:
            return None

    @_check_properties_valid
    def get_gdc_col_description(self):
        if "description" in self.properties:
            return self.properties["description"]
        elif "common" in self.properties:
            return self.properties["common"]["description"]
        return ""

    # Setters & Getters
    def get_schema(self):
        return self.schema

    def set_column_name(self, column_name):
        self.column_name = column_name
        self.candidates = self.get_gdc_candidates()

    def get_column_name(self):
        return self.column_name

    def set_properties(self, properties):
        self.properties = properties

    @_check_properties_valid
    def get_properties(self):
        return self.properties


def load_gdc_schema():
    if not os.path.exists(PATH_TO_GDC_SCHEMA):
        return {}
    with open(PATH_TO_GDC_SCHEMA, "r") as f:
        data = json.load(f)
    return data

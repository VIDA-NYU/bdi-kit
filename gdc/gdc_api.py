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
                logger.error(
                    "Invalid subschema, make sure your subschema is in schema.keys!"
                )
            else:
                self.subschema = subschema

        if column_name is not None:
            self.set_column_name(column_name)
        else:
            self.column_name = None
            self.candidates = None

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
            items = {self.subschema: self.get_schema()[self.subschema]}.items()
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

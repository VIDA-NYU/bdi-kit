import pandas as pd
import pytest
from bdikit.standards.base import BaseStandard

# Sample data for testing
FAKE_DATA = {
    "attr1": {
        "description": "Description for attr1",
        "values": {
            "value1.1": "Description for value1.1",
            "value1.2": "Description for value1.2",
        },
    },
    "attr2": {
        "description": "Description for attr2",
        "values": {
            "value2.1": "Description for value2.1",
            "value2.2": "Description for value2.2",
        },
    },
}


class FakeStandard(BaseStandard):
    def __init__(self, data):
        self.data = data

    def get_attributes(self):
        return list(self.data.keys())

    def get_attribute_values(self, attribute_names):
        return {
            name: list(self.data[name]["values"].keys()) for name in attribute_names
        }

    def get_attribute_metadata(self, attribute_names):
        return {
            name: {
                "description": self.data[name]["description"],
                "value_names": list(self.data[name]["values"].keys()),
                "value_descriptions": list(self.data[name]["values"].values()),
            }
            for name in attribute_names
        }


@pytest.fixture
def fake_standard():
    return FakeStandard(FAKE_DATA)


def test_get_attributes(fake_standard):
    assert fake_standard.get_attributes() == ["attr1", "attr2"]


def test_get_attribute_values(fake_standard):
    expected = {
        "attr1": ["value1.1", "value1.2"],
        "attr2": ["value2.1", "value2.2"],
    }
    assert fake_standard.get_attribute_values(["attr1", "attr2"]) == expected


def test_get_attribute_metadata(fake_standard):
    expected = {
        "attr1": {
            "description": "Description for attr1",
            "value_names": ["value1.1", "value1.2"],
            "value_descriptions": [
                "Description for value1.1",
                "Description for value1.2",
            ],
        },
        "attr2": {
            "description": "Description for attr2",
            "value_names": ["value2.1", "value2.2"],
            "value_descriptions": [
                "Description for value2.1",
                "Description for value2.2",
            ],
        },
    }
    assert fake_standard.get_attribute_metadata(["attr1", "attr2"]) == expected


def test_get_dataframe_rep(fake_standard):
    df = fake_standard._get_dataframe_rep()
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["attr1", "attr2"]
    assert df["attr1"].tolist() == ["value1.1", "value1.2"]
    assert df["attr2"].tolist() == ["value2.1", "value2.2"]

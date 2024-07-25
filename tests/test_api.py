import bdikit as bdi
import pandas as pd
from bdikit.mapping_algorithms.value_mapping.value_mappers import (
    FunctionValueMapper,
    IdentityValueMapper,
)


def test_bdi_match_schema_with_dataframes():
    # given
    source = pd.DataFrame({"column_1": ["a1", "b1", "c1"], "col_2": ["a2", "b2", "c2"]})
    target = pd.DataFrame({"column_1a": ["a1", "b1", "c1"], "col2": ["a2", "b2", "c2"]})

    # when
    df_matches = bdi.match_schema(source, target=target, method="similarity_flooding")

    # then assert that the df_matches contains a row with the value 'column_1'
    # in the column 'source' and the value 'column_1a' in the 'target' value
    assert "source" in df_matches.columns
    assert "target" in df_matches.columns

    df_filter = df_matches["source"] == "column_1"
    assert df_matches[df_filter]["target"].values[0] == "column_1a"

    df_filter = df_matches["source"] == "col_2"
    assert df_matches[df_filter]["target"].values[0] == "col2"


def test_bdi_match_schema_to_gdc():
    # given
    source = pd.DataFrame(
        {
            "FIGO_stage": [
                "Stage 0",
                "Stage I",
                "Stage IA",
                "Stage IA1",
                "Stage IA2",
            ],
            "Ethnicity": [
                "Not-Hispanic or Latino",
                "Hispanic or Latino",
                "Not reported",
                "Hispanic or Latino",
                "Hispanic or Latino",
            ],
        }
    )

    # when
    df_matches = bdi.match_schema(source, target="gdc", method="coma")

    # then df_matches must contain target columns that come from the GDC dictionary
    assert df_matches.empty == False
    assert "source" in df_matches.columns
    assert "target" in df_matches.columns

    df_filter = df_matches["source"] == "Ethnicity"
    assert df_matches[df_filter]["target"].values[0] == "ethnicity"

    df_filter = df_matches["source"] == "FIGO_stage"
    assert df_matches[df_filter]["target"].values[0] == "figo_stage"


def test_bdi_top_matches_with_dataframes():
    # given
    source = pd.DataFrame({"tumor_size": ["a1", "b1", "c1"]})
    target = pd.DataFrame(
        {
            "color": ["red", "blue", "green", "yellow"],
            "tumor_length": [12, 23, 34, 45],
            "tumor_magnitude": [12, 23, 34, 45],
            "tumor_size": [12, 23, 34, 45],
        }
    )

    # when
    df_matches = bdi.top_matches(source, target=target, top_k=3)

    # then
    assert len(df_matches.index) == 3
    assert "source" in df_matches.columns
    assert "target" in df_matches.columns
    assert "similarity" in df_matches.columns

    df_filter = df_matches["source"] == "tumor_size"
    assert "tumor_size" in df_matches[df_filter]["target"].tolist()
    assert "tumor_magnitude" in df_matches[df_filter]["target"].tolist()
    assert "tumor_length" in df_matches[df_filter]["target"].tolist()


def test_bdi_top_matches_gdc():
    # given
    source = pd.DataFrame(
        {
            "FIGO_stage": [
                "Stage 0",
                "Stage I",
                "Stage IA",
                "Stage IA1",
                "Stage IA2",
            ],
            "Ethnicity": [
                "Not-Hispanic or Latino",
                "Hispanic or Latino",
                "Not reported",
                "Hispanic or Latino",
                "Hispanic or Latino",
            ],
        }
    )

    # when
    df_matches = bdi.top_matches(source, target="gdc", top_k=5)

    # then
    assert df_matches.empty == False
    assert "source" in df_matches.columns
    assert "target" in df_matches.columns
    assert "similarity" in df_matches.columns

    df_filter = df_matches["source"] == "FIGO_stage"
    assert len(df_matches[df_filter]) == 5
    assert "figo_stage" in df_matches[df_filter]["target"].tolist()
    assert "uicc_clinical_stage" in df_matches[df_filter]["target"].tolist()

    df_filter = df_matches["source"] == "Ethnicity"
    assert len(df_matches[df_filter]) == 5
    assert "ethnicity" in df_matches[df_filter]["target"].tolist()
    assert "race" in df_matches[df_filter]["target"].tolist()


def test_map_dataframe_column_values():
    # given
    str_column_1 = ["a", "b", "c", "d", "e"]
    str_column_2 = ["a", "b", "c", "d", "e"]
    df_base = pd.DataFrame({"column_str_1": str_column_1, "column_str_2": str_column_2})

    value_mapping_spec = [
        {
            "source": "column_str_1",
            "target": "string column 1",
            "mapper": IdentityValueMapper(),
        },
        {
            "source": "column_str_2",
            "target": "string column 2",
            "mapper": FunctionValueMapper(function=lambda x: x.upper()),
        },
    ]

    # when
    df_mapped = bdi.materialize_mapping(df_base, mapping_spec=value_mapping_spec)

    # then
    assert len(df_mapped.columns) == 2

    assert "string column 1" in df_mapped.columns
    assert df_mapped["string column 1"].eq(str_column_1).all()

    assert "string column 2" in df_mapped.columns
    assert df_mapped["string column 2"].eq(["A", "B", "C", "D", "E"]).all()


def test_value_mapping_dataframe():
    # given
    df_source = pd.DataFrame(
        {"src_column": ["Red Apple", "Banana", "Oorange", "Strawberry"]}
    )
    df_target = pd.DataFrame(
        {"tgt_column": ["apple", "banana", "orange", "kiwi", "grapes"]}
    )

    df_matches = pd.DataFrame({"source": ["src_column"], "target": ["tgt_column"]})

    # when
    value_mappings = bdi.match_values(df_source, df_target, df_matches, method="tfidf")

    # then
    assert len(value_mappings) == 1
    mapping = value_mappings[0]
    assert mapping is not None
    assert isinstance(mapping, pd.DataFrame)
    assert mapping.attrs["source"] == "src_column"
    assert mapping.attrs["target"] == "tgt_column"
    assert len(mapping) == len(df_source)


def test_end_to_end_api_integration():
    # given
    df_source = pd.DataFrame(
        {"src_column": ["Red Apple", "Banana", "Oorange", "Strawberry"]}
    )
    df_target = pd.DataFrame(
        {"tgt_column": ["apple", "banana", "orange", "kiwi", "grapes"]}
    )

    # when
    column_mappings = bdi.match_schema(df_source, df_target, method="coma")
    # then
    assert column_mappings is not None
    assert column_mappings.empty == False
    assert "source" in column_mappings.columns
    assert "target" in column_mappings.columns
    assert len(column_mappings.index) == 1

    # when: pass output of match_schema() directly to materialize_mapping(),
    # the column must be ranamed to the target column without any value mapping
    df_mapped = bdi.materialize_mapping(df_source, column_mappings)
    # then
    assert "tgt_column" in df_mapped.columns
    assert df_mapped["tgt_column"].tolist() == [
        "Red Apple",
        "Banana",
        "Oorange",
        "Strawberry",
    ]

    # when: we pass the output of match_schema()
    value_mappings = bdi.match_values(
        df_source, df_target, column_mappings, method="tfidf"
    )

    # then: a list of value matches must be computed
    assert len(value_mappings) == 1
    mapping = value_mappings[0]
    assert mapping is not None
    assert isinstance(mapping, pd.DataFrame)
    assert len(mapping) == len(df_source)
    assert mapping.attrs["source"] == "src_column"
    assert mapping.attrs["target"] == "tgt_column"

    # when: pass output of match_values() to materialize_mapping(),
    df_mapped = bdi.materialize_mapping(df_source, value_mappings)

    # then: the column must be ranamed and values must be mapped to the
    # matching values found during the value matching step
    assert isinstance(df_mapped, pd.DataFrame)
    assert "tgt_column" in df_mapped.columns
    assert df_mapped["tgt_column"].tolist() == ["apple", "banana", "orange", None]

    # when: pass output of match_values() to merge_mappings() and then to
    # materialize_mapping()
    harmonization_spec = bdi.merge_mappings(value_mappings, [])
    df_mapped = bdi.materialize_mapping(df_source, harmonization_spec)

    # then: the column must be ranamed and values must be mapped
    assert isinstance(df_mapped, pd.DataFrame)
    assert "tgt_column" in df_mapped.columns
    assert df_mapped["tgt_column"].tolist() == ["apple", "banana", "orange", None]

    # when: user mappings are specified in merge_mappings()
    user_mappings = [
        {
            "source": "src_column",
            "target": "tgt_column",
            "matches": [
                ("Red Apple", "APPLE"),
                ("Banana", "BANANA"),
                ("Oorange", "ORANGE"),
            ],
        }
    ]
    harmonization_spec = bdi.merge_mappings(value_mappings, user_mappings)
    df_mapped = bdi.materialize_mapping(df_source, harmonization_spec)

    # then: user mappings take precedence, so the column must be ranamed and
    # values must be mapped according the provide user_mappings
    assert "tgt_column" in df_mapped.columns
    assert df_mapped["tgt_column"].tolist() == ["APPLE", "BANANA", "ORANGE", None]


def test_top_matches_and_match_values_integration():
    # given
    df_source = pd.DataFrame(
        {"fruits": ["Red Apple", "Banana", "Oorange", "Strawberry"]}
    )
    df_target = pd.DataFrame(
        {
            "fruit_types": ["apple", "banana", "orange", "kiwi", "grapes"],
            "fruit_names": ["apple", "banana", "melon", "kiwi", "grapes"],
            "fruit_id": ["1", "2", "3", "4", "5"],
        }
    )

    # when
    df_matches = bdi.top_matches(df_source, target=df_target)

    # then
    assert len(df_matches.index) == 3
    assert "source" in df_matches.columns
    assert "target" in df_matches.columns
    assert "similarity" in df_matches.columns

    # when
    df_matches = bdi.match_values(
        df_source, df_target, column_mapping=df_matches, method="tfidf"
    )
    assert isinstance(df_matches, list)
    assert len(df_matches) == 3
    for df in df_matches:
        assert isinstance(df, pd.DataFrame)
        assert "source" in df.columns
        assert "target" in df.columns
        assert "similarity" in df.columns
        assert df.attrs["source"] == "fruits"
        assert df.attrs["target"] in ["fruit_types", "fruit_names", "fruit_id"]

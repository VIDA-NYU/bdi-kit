import bdikit as bdi
import pandas as pd


def test_bdi_match_columns_with_dataframes():
    # given
    source = pd.DataFrame({"column_1": ["a1", "b1", "c1"], "col_2": ["a2", "b2", "c2"]})
    target = pd.DataFrame({"column_1a": ["a1", "b1", "c1"], "col2": ["a2", "b2", "c2"]})

    # when
    df_matches = bdi.match_columns(source, target=target, method="similarity_flooding")

    # then assert that the df_matches contains a row with the value 'column_1'
    # in the column 'source' and the value 'column_1a' in the 'target' value
    assert "source" in df_matches.columns
    assert "target" in df_matches.columns

    df_filter = df_matches["source"] == "column_1"
    assert df_matches[df_filter]["target"].values[0] == "column_1a"

    df_filter = df_matches["source"] == "col_2"
    assert df_matches[df_filter]["target"].values[0] == "col2"


def test_bdi_match_columns_to_gdc():
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
    df_matches = bdi.match_columns(source, target="gdc", method="coma")

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
    assert "matches" in df_matches.columns
    assert "similarity" in df_matches.columns

    df_filter = df_matches["source"] == "tumor_size"
    assert "tumor_size" in df_matches[df_filter]["matches"].tolist()
    assert "tumor_magnitude" in df_matches[df_filter]["matches"].tolist()
    assert "tumor_length" in df_matches[df_filter]["matches"].tolist()


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
    assert "matches" in df_matches.columns
    assert "similarity" in df_matches.columns

    df_filter = df_matches["source"] == "FIGO_stage"
    assert len(df_matches[df_filter]) == 5
    assert "figo_stage" in df_matches[df_filter]["matches"].tolist()
    assert "uicc_clinical_stage" in df_matches[df_filter]["matches"].tolist()

    df_filter = df_matches["source"] == "Ethnicity"
    assert len(df_matches[df_filter]) == 5
    assert "ethnicity" in df_matches[df_filter]["matches"].tolist()
    assert "race" in df_matches[df_filter]["matches"].tolist()

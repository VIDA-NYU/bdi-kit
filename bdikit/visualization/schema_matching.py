import hashlib
import json
import logging
from datetime import datetime
from os.path import dirname, exists, join
from typing import Any, Dict, List, Optional, Tuple, Union

import altair as alt
import datamart_profiler
import numpy as np
import pandas as pd
import panel as pn
from openai import OpenAI
from polyfuzz import PolyFuzz
from polyfuzz.models import RapidFuzz
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

from bdikit.download import BDIKIT_CACHE_DIR
from bdikit.mapping_algorithms.column_mapping.topk_matchers import (
    CLTopkColumnMatcher,
    ColumnScore,
    TopkColumnMatcher,
    TopkMatching,
)
from bdikit.models.contrastive_learning.cl_api import DEFAULT_CL_MODEL
from bdikit.utils import get_gdc_layered_metadata, read_gdc_schema

GDC_DATA_PATH = join(dirname(__file__), "../resource/gdc_table.csv")

# Schema.org types
SCHEMA_ENUMERATION = "http://schema.org/Enumeration"
SCHEMA_TEXT = "http://schema.org/Text"
SCHEMA_FLOAT = "http://schema.org/Float"
SCHEMA_INTEGER = "http://schema.org/Integer"
SCHEMA_BOOLEAN = "http://schema.org/Boolean"


logger = logging.getLogger("bdiviz")

pn.extension("tabulator")  # type: ignore
pn.extension("mathjax")  # type: ignore
pn.extension("vega")  # type: ignore
pn.extension("floatpanel")  # type: ignore


def truncate_text(text: str, max_chars: int):
    if len(text) > max_chars:
        return text[:max_chars] + "..."
    else:
        return text


class BDISchemaMatchingHeatMap(TopkColumnMatcher):
    def __init__(
        self,
        source: pd.DataFrame,
        additional_sources: Optional[
            Dict[str, pd.DataFrame]
        ] = None,  # {"Dou": dou_df, ...}
        target: Union[pd.DataFrame, str] = "gdc",
        top_k: int = 10,
        heatmap_recommendations: Optional[List[Dict]] = None,
        max_chars_samples: int = 150,
        height: int = 600,
        ai_assitant: bool = False,
    ) -> None:
        self.json_path = "heatmap_recommendations.json"

        # Sources color palette
        self.source_prefix = ""
        self.source_colors = [
            "#ffa600",
            "#ff6361",
            "#bc5090",
            "#58508d",
            "#003f5c",
        ]

        # Source and target data
        self.source = source.sample(min(1000, len(source))).reset_index(drop=True)
        self.additional_sources = (
            {
                name: df.sample(min(1000, len(df))).reset_index(drop=True)
                for name, df in additional_sources.items()
            }
            if additional_sources
            else None
        )
        self.target = target

        # Source columns lookup
        self.source_columns = [
            (column, self.source_prefix) for column in source.columns
        ]
        if self.additional_sources:
            for name, df in self.additional_sources.items():
                self.source_columns.extend([(column, name) for column in df.columns])

        self.top_k = max(1, min(top_k, 40))

        self.rec_table_df = None
        self.rec_list_df = None
        self.rec_cols = None
        self.subschemas = None
        self.clusters = None

        # Selected column
        self.selected_row = None

        # Embeddings
        self.l_features = {}
        self.r_features = None

        # Load cached results
        cached_heatmap_recommendations = self._load_cached_results()
        if cached_heatmap_recommendations is not None:
            self.heatmap_recommendations = cached_heatmap_recommendations
        else:
            if heatmap_recommendations is None:
                self.heatmap_recommendations = self._generate_top_k_matches()
            else:
                self.heatmap_recommendations = heatmap_recommendations
            self._cache_results(self.heatmap_recommendations)

        self._write_json(self.heatmap_recommendations)

        self.candidates_dfs = self._clean_heatmap_recommendations()

        self.height = height

        # Undo/Redo
        # The undo/redo stack is a list of data
        # Data is like this: {'Candidate column': 'Country', 'Top k columns': [['country_of_birth', '0.5726'], ...]}
        self.undo_stack = []
        self.redo_stack = []
        self.logs = []

        self._get_heatmap()
        self._gen_clusters()

        # Value matches
        self.value_matches_dfs = self._generate_all_value_matches()

        # AI Assistant
        self.assistant = None
        if ai_assitant:
            self.assistant = LLMAssistant()
            self.chat_history = []

    def _generate_top_k_matches(self) -> List[Dict]:
        if isinstance(self.target, pd.DataFrame):
            target_df = self.target
        elif self.target == "gdc":
            target_df = pd.read_csv(GDC_DATA_PATH)
        else:
            raise ValueError("Invalid target value. Must be a DataFrame or 'gdc'.")

        topk_matcher = CLTopkColumnMatcher(model_name=DEFAULT_CL_MODEL)

        # Cache features
        self.l_features[self.source_prefix] = topk_matcher.api.get_embeddings(
            self.source
        )
        self.r_features = topk_matcher.api.get_embeddings(target_df)

        top_k_matches = self._generate_top_k_matches_from_embeddings(
            self.source, target_df, self.l_features[self.source_prefix], self.r_features
        )

        output_json = []
        for match in top_k_matches:
            source_dict = {
                "source_column": match["source_column"],
                "top_k_columns": [],
                "source_dataset": self.source_prefix,
            }
            for column in match["top_k_columns"]:
                source_dict["top_k_columns"].append(
                    [column.column_name, float(column.score)]
                )
            output_json.append(source_dict)

        if self.additional_sources:
            for name, df in self.additional_sources.items():
                self.l_features[name] = topk_matcher.api.get_embeddings(df)
                additional_top_k_matches = self._generate_top_k_matches_from_embeddings(
                    df, target_df, self.l_features[name], self.r_features
                )
                for match in additional_top_k_matches:
                    source_dict = {
                        "source_column": match["source_column"],
                        "top_k_columns": [],
                        "source_dataset": name,
                    }
                    for column in match["top_k_columns"]:
                        source_dict["top_k_columns"].append(
                            [column.column_name, float(column.score)]
                        )
                    output_json.append(source_dict)

        return output_json

    def _generate_top_k_matches_from_embeddings(
        self,
        source: pd.DataFrame,
        target: pd.DataFrame,
        l_features: List[np.ndarray],
        r_features: List[np.ndarray],
    ) -> List[Dict]:
        sim = cosine_similarity(l_features, r_features)  # type: ignore

        top_k_results = []
        for index, similarities in enumerate(sim):
            top_k_indices = np.argsort(similarities)[::-1][: self.top_k]
            top_k_columns = [
                ColumnScore(column_name=target.columns[i], score=similarities[i])
                for i in top_k_indices
            ]
            top_k_results.append(
                {
                    "source_column": source.columns[index],
                    "top_k_columns": top_k_columns,
                }
            )

        return top_k_results

    def _clean_heatmap_recommendations(self):
        candidates_dfs = {}
        if not isinstance(self.target, pd.DataFrame) and self.target == "gdc":
            gdc_metadata = get_gdc_layered_metadata()
            for column_data in self.heatmap_recommendations:
                column_name = column_data["source_column"]
                if column_name not in candidates_dfs:
                    recommendations = []
                else:
                    recommendations = candidates_dfs[column_name].values.tolist()
                for candidate_name, candidate_similarity in column_data[
                    "top_k_columns"
                ]:
                    subschema, gdc_data = gdc_metadata[candidate_name]
                    candidate_description = gdc_data.get("description", "")
                    candidate_type = self._gdc_get_column_type(gdc_data)
                    candidate_values = ", ".join(gdc_data.get("enum", []))
                    # candidate_values = truncate_text(candidate_values, max_chars_samples)
                    recommendations.append(
                        [
                            candidate_name,
                            candidate_similarity,
                            candidate_values,
                            candidate_type,
                            candidate_description,
                            subschema,
                        ]
                    )
                candidates_dfs[column_name] = pd.DataFrame(
                    recommendations,
                    columns=[
                        "Candidate",
                        "Similarity",
                        "Values (sample)",
                        "Type",
                        "Description",
                        "Subschema",
                    ],
                )
        else:
            profiled_data = datamart_profiler.process_dataset(
                self.target, coverage=False, indexes=False
            )["columns"]
            for column_data in self.heatmap_recommendations:
                column_name = column_data["source_column"]
                recommendations = []
                for candidate_name, candidate_similarity in column_data[
                    "top_k_columns"
                ]:
                    # check candidate type generated by profiler
                    profiled_cand = next(
                        profiled_cand
                        for profiled_cand in profiled_data
                        if profiled_cand["name"] == candidate_name
                    )
                    if SCHEMA_ENUMERATION in profiled_cand["semantic_types"]:
                        candidate_type = "enum"
                    elif SCHEMA_BOOLEAN in profiled_cand["semantic_types"]:
                        candidate_type = "boolean"
                    elif (
                        SCHEMA_FLOAT in profiled_cand["structural_type"]
                        or SCHEMA_INTEGER in profiled_cand["structural_type"]
                    ):
                        candidate_type = "number"
                    else:
                        candidate_type = "string"

                    candidate_values = ", ".join(
                        self.target[candidate_name].astype(str).unique()
                    )
                    recommendations.append(
                        (
                            candidate_name,
                            candidate_similarity,
                            candidate_values,
                            candidate_type,
                        )
                    )

                candidates_dfs[column_name] = pd.DataFrame(
                    recommendations,
                    columns=[
                        "Candidate",
                        "Similarity",
                        "Values (sample)",
                        "Type",
                    ],
                )
        return candidates_dfs

    def _load_json(self) -> "List[Dict] | None":
        cache_path = join(
            BDIKIT_CACHE_DIR,
            f"reducings_{self._get_ground_truth_checksum()}_{self._get_data_checksum()}_{self.top_k}.tmp.json",
        )
        if exists(cache_path):
            with open(cache_path) as f:
                data = json.load(f)
                return data
        return None

    def _write_json(self, data: List[Dict]) -> None:
        self.heatmap_recommendations = data

        # cache_path = join(
        #     BDIKIT_CACHE_DIR,
        #     f"reducings_{self._get_ground_truth_checksum()}_{self._get_data_checksum()}_{self.top_k}.tmp.json",
        # )

        # with open(cache_path, "w") as f:
        #     json.dump(data, f)

    def _get_heatmap(self) -> None:
        recommendations = self.heatmap_recommendations
        rec_cols = set()
        rec_table = []
        rec_list = []

        for d in recommendations:
            col_dict = {"Column": d["source_column"]}
            for c in d["top_k_columns"]:
                rec_cols.add(c[0])
                col_dict[c[0]] = c[1]
                rec_row = {
                    "Column": d["source_column"],
                    "Recommendation": c[0],
                    "Value": c[1],
                    "DataFrame": d["source_dataset"],
                }
                # [GDC] get description and values
                if not isinstance(self.target, pd.DataFrame) and self.target == "gdc":
                    candidates_info = self.candidates_dfs[d["source_column"]]
                    candidate_info = candidates_info[
                        candidates_info["Candidate"] == c[0]
                    ]
                    rec_row["Description"] = candidate_info["Description"].values[0]
                    rec_row["Values (sample)"] = candidate_info[
                        "Values (sample)"
                    ].values[0]
                    rec_row["Subschema"] = candidate_info["Subschema"].values[0]
                rec_list.append(rec_row)
            rec_table.append(col_dict)

        rec_cols = list(rec_cols)
        rec_cols.sort()

        rec_table_df = pd.DataFrame(rec_table)
        rec_list_df = pd.DataFrame(rec_list)
        rec_list_df["Value"] = pd.to_numeric(rec_list_df["Value"])
        rec_list_df["DataFrame"] = rec_list_df["DataFrame"].astype(str)

        self.rec_table_df = rec_table_df
        self.rec_list_df = rec_list_df
        self.rec_cols = rec_cols

        # [GDC] get subschema information
        if not isinstance(self.target, pd.DataFrame) and self.target == "gdc":
            self.get_cols_subschema()

    def _gen_clusters(self) -> None:
        knn = NearestNeighbors(n_neighbors=5)
        l_features_flat = []
        for _, l_features in self.l_features.items():
            l_features_flat.extend(l_features)
        knn.fit(np.array(l_features_flat))
        clusters_idx = [
            knn.kneighbors([l_feature], return_distance=False)[0]
            for l_feature in self.l_features[self.source_prefix]
        ]

        clusters = {}
        for i, source_column in enumerate(self.source.columns):
            cluster_idx = clusters_idx[i]
            cluster = []
            for idx in cluster_idx:
                cluster.append(self.source_columns[idx])
            clusters[source_column] = cluster
        self.clusters = clusters

    def get_cols_subschema(self) -> None:
        subschemas = []
        schema = read_gdc_schema()
        for parent, values in schema.items():
            for candidate in values["properties"].keys():
                if candidate in self.rec_cols:
                    if parent not in subschemas:
                        subschemas.append(parent)

        self.subschemas = subschemas

    def _gdc_get_column_type(self, properties: Dict) -> "str | None":
        if "enum" in properties:
            return "enum"
        elif "type" in properties:
            return properties["type"]
        else:
            return None

    def _gdc_get_column_description(self, properties: Dict) -> str:
        if "description" in properties:
            return properties["description"]
        elif "common" in properties:
            return properties["common"]["description"]
        return ""

    def _gdc_get_column_values(self, properties: Dict) -> "List[str] | None":
        col_type = self._gdc_get_column_type(properties)
        if col_type == "enum":
            return properties["enum"]
        elif col_type == "number" or col_type == "integer" or col_type == "float":
            return [
                str(properties["minimum"]) if "minimum" in properties else "-inf",
                str(properties["maximum"]) if "maximum" in properties else "inf",
            ]
        elif col_type == "boolean":
            return ["True", "False"]
        else:
            return None

    def _generate_all_value_matches(self):
        value_matches_dfs = {}
        rapidfuzz_matcher = RapidFuzz(n_jobs=1)
        value_matcher = PolyFuzz(rapidfuzz_matcher)

        for source_column, source_df in self.source_columns:
            if source_df == self.source_prefix:
                if pd.api.types.is_numeric_dtype(self.source[source_column]):
                    continue

                source_values = list(self.source[source_column].dropna().unique())[:20]

            elif self.additional_sources and source_df in self.additional_sources:
                if pd.api.types.is_numeric_dtype(
                    self.additional_sources[source_df][source_column]
                ):
                    continue

                source_values = list(
                    self.additional_sources[source_df][source_column].dropna().unique()
                )[:20]

            else:
                continue

            value_comparison = {
                "Source Value": source_values,
            }

            for _, row in self.candidates_dfs[source_column].iterrows():
                target_values = row["Values (sample)"].split(", ")

                value_matcher.match(source_values, target_values)
                match_results = value_matcher.get_matches()

                value_comparison[row["Candidate"]] = list(match_results["To"])

            value_matches_dfs[source_column] = pd.DataFrame(
                dict([(k, pd.Series(v)) for k, v in value_comparison.items()])
            ).fillna("")

        return value_matches_dfs

    def _accept_match(self) -> None:
        if self.selected_row is None:
            return
        col_name = self.selected_row["Column"].values[0]
        match_name = self.selected_row["Recommendation"].values[0]
        col_source_df = self.selected_row["DataFrame"].values[0]
        recommendations = self.heatmap_recommendations
        for idx, d in enumerate(recommendations):
            candidate_name = d["source_column"]
            if candidate_name != col_name:
                continue
            for top_k_name, top_k_score in d["top_k_columns"]:
                if top_k_name == match_name and col_source_df == d["source_dataset"]:
                    recommendations[idx] = {
                        "source_column": candidate_name,
                        "top_k_columns": [[top_k_name, top_k_score]],
                        "source_dataset": col_source_df,
                    }

                    # record the action
                    self._record_user_action("accept", d)
                    self._record_log("accept", candidate_name, top_k_name)

                    self._write_json(recommendations)
                    self._get_heatmap()
                    return

    def _reject_match(self) -> None:
        if self.selected_row is None:
            return
        col_name = self.selected_row["Column"].values[0]
        match_name = self.selected_row["Recommendation"].values[0]
        col_source_df = self.selected_row["DataFrame"].values[0]
        recommendations = self.heatmap_recommendations
        for idx, d in enumerate(recommendations):
            candidate_name = d["source_column"]
            if candidate_name != col_name:
                continue
            new_top_k = []
            for top_k_name, top_k_score in d["top_k_columns"]:
                if top_k_name != match_name or col_source_df != d["source_dataset"]:
                    new_top_k.append([top_k_name, top_k_score])
            recommendations[idx] = {
                "source_column": candidate_name,
                "top_k_columns": new_top_k,
                "source_dataset": col_source_df,
            }

            # record the action
            self._record_user_action("reject", d)
            self._record_log("reject", candidate_name, match_name)

            self._write_json(recommendations)
            self._get_heatmap()
            return

    def _discard_column(self, select_column: Optional[str]) -> None:
        if not select_column and select_column not in self.source.columns:
            logger.critical(f"Invalid column: {select_column}")
            return

        logger.critical(f"Discarding column: {select_column}")
        recommendations = self.heatmap_recommendations
        for idx, d in enumerate(recommendations):
            candidate_name = d["source_column"]
            if candidate_name == select_column:
                recommendations.pop(idx)
                self._write_json(recommendations)
                self._record_user_action("discard", d)
                self._record_log("discard", candidate_name, "")
                self._get_heatmap()
                return

    def _plot_heatmap_base(self, heatmap_rec_list: pd.DataFrame) -> pn.pane.Vega:
        single = alt.selection_point(name="single")

        tooltip = [
            alt.Tooltip("Column", title="Column"),
            alt.Tooltip("Recommendation", title="Recommendation"),
            alt.Tooltip("Value", title="Similarity"),
        ]
        # facet = alt.Facet(alt.Undefined)

        base = (
            alt.Chart(heatmap_rec_list)
            .transform_calculate(
                Column=alt.datum["DataFrame"] + ">" + alt.datum["Column"]
            )
            .encode(
                y=alt.Y("Column:O", sort=None),
                x=alt.X("Recommendation:O", sort=None).axis(labelAngle=-45),
                color=alt.condition(
                    single,
                    alt.Color("Value:Q").scale(domainMax=1, domainMin=0),
                    alt.value("lightgray"),
                ),  # type: ignore
                tooltip=tooltip,
            )
            .add_params(single)
        )
        background = base.mark_rect(size=100)

        box_sources = []
        if self.additional_sources:
            y_source = heatmap_rec_list[
                heatmap_rec_list["DataFrame"] == self.source_prefix
            ]
            box_source_base = (
                alt.Chart(y_source)
                .transform_calculate(
                    Column=alt.datum["DataFrame"] + ">" + alt.datum["Column"]
                )
                .encode(
                    text=alt.condition(
                        alt.datum["DataFrame"] == self.source_prefix,
                        alt.value("source"),
                        alt.TextDatum("DataFrame"),
                    ),  # type: ignore
                    y="Column:O",
                )
            )
            box_source = box_source_base.mark_rect(
                color="", stroke="gray", strokeWidth=1
            )
            box_source_text = box_source.mark_text(baseline="middle", color="gray")
            box_sources.append(box_source)
            box_sources.append(box_source_text)

            for idx, (name, _) in enumerate(self.additional_sources.items()):
                color = self.source_colors[idx]

                y_source = heatmap_rec_list[heatmap_rec_list["DataFrame"] == name]
                box_source_base = (
                    alt.Chart(y_source)
                    .transform_calculate(
                        Column=alt.datum["DataFrame"] + ">" + alt.datum["Column"]
                    )
                    .encode(
                        text="DataFrame:O",
                        y="Column:O",
                    )
                )
                box_source = box_source_base.mark_rect(
                    color="", stroke=color, strokeWidth=1
                )
                box_source_text = box_source.mark_text(baseline="middle", color=color)
                box_sources.append(box_source)
                box_sources.append(box_source_text)

        # rule1 = background.mark_rect(color="", stroke="orange", strokeWidth=2).transform_filter(alt.FieldEqualPredicate(field='DataFrame', equal='source'))
        # rule2 = background.mark_rect(color="", stroke="yellow", strokeWidth=2).transform_filter(alt.FieldOneOfPredicate(field='DataFrame', oneOf=list(self.additional_sources.keys())))
        return pn.pane.Vega(alt.layer(background, *box_sources))

    def _update_column_selection(
        self, heatmap_rec_list: pd.DataFrame, selection: List[int]
    ) -> Tuple[str, str, str]:
        selected_idx = [selection[0] - 1]
        selected_row = heatmap_rec_list.iloc[selected_idx]
        self.selected_row = selected_row
        column = selected_row["Column"].values[0]
        rec = selected_row["Recommendation"].values[0]
        source_df = selected_row["DataFrame"].values[0]
        return column, rec, source_df

    def _gdc_candidates_info(
        self, heatmap_rec_list: pd.DataFrame, selection: List[int], n_samples: int = 20
    ) -> pn.pane.Markdown:
        if not selection:
            return pn.pane.Markdown(
                """
                
                ### Selected Recommendation
                
                *No selection.*

            """
            )
        column, rec, _ = self._update_column_selection(heatmap_rec_list, selection)
        if not isinstance(self.target, pd.DataFrame) and self.target == "gdc":
            df = self.candidates_dfs[column][
                self.candidates_dfs[column]["Candidate"] == rec
            ]

            rec_rank = df.index[0]
            values = df["Values (sample)"].values[0].split(", ")

            sample = "\n\n"
            for i, v in enumerate(values[:n_samples]):
                sample += f"""            - {v}\n"""
            if len(values) == 0:
                sample = "*No values provided.*"
            is_sample = f" ({n_samples} samples)" if len(values) > n_samples else ""

            descrip = df.loc[rec_rank, "Description"]
            if len(df.loc[rec_rank, "Description"]) == 0:
                descrip = "*No description provided.*"

            rec_info = f"""
            ### Selected Recommendation

            **Name:** {rec}

            **Rank:** {rec_rank+1}

            **Similarity:** {df.loc[rec_rank,'Similarity']}

            **Subschema:** {df.loc[rec_rank,'Subschema']}

            **Description:** {descrip}

            **Values{is_sample}:** {sample}

        """
            rec_pane = pn.pane.Markdown(rec_info)
            return rec_pane
        else:
            return pn.pane.Markdown(
                "GDC candidates info is not available for this target."
            )

    def _candidates_table(
        self, heatmap_rec_list: pd.DataFrame, selection: List[int]
    ) -> pn.widgets.Tabulator:
        if not selection:
            return pn.pane.Markdown("## No selection")
        column, rec, _ = self._update_column_selection(heatmap_rec_list, selection)
        df = self.candidates_dfs[column][
            self.candidates_dfs[column]["Candidate"] == rec
        ]

        bokeh_formatters = {
            #'Similarity': {'type': 'progress', 'min': 0.0, 'max': 1.0, 'legend': True}, # Show similarity as bars - Not working properly
            "Description": {"type": "textarea"},
            "Values (sample)": {"type": "textarea"},
        }
        text_align = {"Similarity": "center", "index": "center"}
        widths = {
            "index": "7%",
            "Candidate": "20%",
            "Similarity": "10%",
            "Description": "33%",
            "Values (sample)": "30%",
        }

        table_candidates = pn.widgets.Tabulator(
            df,
            formatters=bokeh_formatters,
            text_align=text_align,
            widths=widths,
            sizing_mode="stretch_width",
            embed_content=True,
            header_align="center",
            disabled=True,
            theme="bootstrap5",
            theme_classes=["thead-dark", "table-sm"],
        )
        return table_candidates

    def _plot_column_histogram(
        self, column: str, dataset: pd.DataFrame
    ) -> "pn.pane.Markdown | alt.LayerChart":
        if pd.api.types.is_numeric_dtype(dataset[column]):
            x = alt.Y(column, bin=True).axis(labelAngle=-45)
            text_color = "transparent"
        else:
            values = list(dataset[column].unique())
            if len(values) == len(dataset[column]):
                string = f"""Values are unique. 
                Some samples: {values[:5]}"""
                return pn.pane.Markdown(string)
            else:
                if np.nan in values:
                    values.remove(np.nan)
                values.sort()
                x = alt.Y(
                    column + ":N",
                    sort=values,
                ).axis(
                    None
                )  # .axis(labelAngle=-45)
            text_color = "black"

        chart = (
            alt.Chart(dataset.fillna("Null"), height=300)
            .mark_bar()
            .encode(
                x="count()",
                y=x,
            )
        )
        text = (
            alt.Chart(dataset.fillna("Null"), height=300)
            .mark_text(color=text_color, fontWeight="bold", fontSize=12, align="right")
            .encode(x="count()", y=x, text=alt.Text(column))
        )
        layered = (
            alt.layer(chart, text)
            .properties(width="container", title="Histogram of " + column)
            .configure(background="#f5f5f5")
        )
        return layered

    def _plot_source_histogram(
        self, source_column: str, heatmap_rec_list: pd.DataFrame, selection: List[int]
    ) -> "pn.pane.Markdown | alt.LayerChart":
        if not selection:
            return self._plot_column_histogram(source_column, self.source)

        column, _, source_df = self._update_column_selection(
            heatmap_rec_list, selection
        )

        if source_df == self.source_prefix:
            return self._plot_column_histogram(column, self.source)
        elif self.additional_sources and source_df in self.additional_sources:
            return self._plot_column_histogram(
                column, self.additional_sources[source_df]
            )
        else:
            return pn.pane.Markdown("No source data found.")

    def _plot_target_histogram(
        self, heatmap_rec_list: pd.DataFrame, selection: List[int]
    ) -> "pn.pane.Markdown | alt.LayerChart":
        if not isinstance(self.target, pd.DataFrame):
            return pn.pane.Markdown("No ground truth provided.")
        if not selection:
            return pn.pane.Markdown("## No selection")

        _, rec, _ = self._update_column_selection(heatmap_rec_list, selection)

        return self._plot_column_histogram(rec, self.target)

    def _plot_value_comparisons(
        self, source_column: str, heatmap_rec_list: pd.DataFrame, selection: List[int]
    ) -> "pn.widgets.Tabulator | pn.pane.Markdown":
        if not selection:
            column = source_column
            rec = None
        else:
            column, rec, _ = self._update_column_selection(heatmap_rec_list, selection)

        if column not in self.value_matches_dfs:
            return pn.pane.Markdown("No value matches found.")

        value_comparisons = self.value_matches_dfs[column]

        value_comparisons = value_comparisons[
            ["Source Value"]
            + list(
                heatmap_rec_list[heatmap_rec_list["Column"] == column]["Recommendation"]
            )
        ]

        frozen_columns = ["Source Value"]
        if rec:
            frozen_columns.append(rec)

        return pn.widgets.Tabulator(
            pd.DataFrame(
                dict([(k, pd.Series(v)) for k, v in value_comparisons.items()])
            ).fillna(""),
            frozen_columns=frozen_columns,
            show_index=False,
            width=700,
            height=200,
        )

    def _plot_pane(
        self,
        select_column: Optional[str] = None,
        select_candidate_type: str = "All",
        subschemas: List[str] = [],
        n_similar: int = 0,
        threshold: float = 0.5,
        acc_click: int = 0,
        rej_click: int = 0,
        discard_click: int = 0,
        undo_click: int = 0,
        redo_click: int = 0,
    ) -> pn.Column:
        heatmap_rec_list = self.rec_list_df[self.rec_list_df["Value"] >= threshold]
        if select_column:
            clustered_tuples = []
            clustered_cols = []
            for cluster_key, cluster_tuples in self.clusters.items():
                if cluster_key == select_column:
                    clustered_tuples = cluster_tuples[: n_similar + 1]
                    clustered_cols = [col for col, _ in clustered_tuples]

            heatmap_rec_list = heatmap_rec_list[
                heatmap_rec_list["Column"].isin(clustered_cols)
            ]

        candidates_df = self.candidates_dfs[select_column]

        def _filter_datatype(heatmap_rec: pd.Series) -> bool:
            if (
                candidates_df[
                    candidates_df["Candidate"] == heatmap_rec["Recommendation"]
                ]["Type"]
                == select_candidate_type
            ).any():
                return True
            else:
                return False

        if select_candidate_type != "All":
            heatmap_rec_list = heatmap_rec_list[
                heatmap_rec_list.apply(_filter_datatype, axis=1)
            ]

        if subschemas:
            subschema_rec_cols = candidates_df[
                candidates_df["Subschema"].isin(subschemas)
            ]["Candidate"].to_list()
            heatmap_rec_list = heatmap_rec_list[
                heatmap_rec_list["Recommendation"].isin(subschema_rec_cols)
            ]

        heatmap_pane = self._plot_heatmap_base(heatmap_rec_list)

        if not isinstance(self.target, pd.DataFrame) and self.target == "gdc":
            cand_info = pn.bind(
                self._gdc_candidates_info,
                heatmap_rec_list,
                heatmap_pane.selection.param.single,
            )
        else:
            cand_info = pn.bind(
                self._plot_target_histogram,
                heatmap_rec_list,
                heatmap_pane.selection.param.single,
            )

        column_hist = pn.bind(
            self._plot_source_histogram,
            select_column,
            heatmap_rec_list,
            heatmap_pane.selection.param.single,
        )

        plot_history = self._plot_history()

        value_comparisons = pn.bind(
            self._plot_value_comparisons,
            select_column,
            heatmap_rec_list,
            heatmap_pane.selection.param.single,
        )

        return pn.Column(
            pn.FloatPanel(
                plot_history,
                name="Operation Logs",
                width=500,
                align="end",
                theme="secondary",
            ),
            pn.Row(
                heatmap_pane,
                scroll=True,
                width=1200,
                styles=dict(background="WhiteSmoke"),
            ),
            pn.Spacer(height=5),
            pn.Card(
                value_comparisons,
                title="Value Comparisons",
                styles=dict(background="WhiteSmoke"),
                scroll=True,
            ),
            pn.Card(
                pn.Row(
                    pn.Column(column_hist, width=500),
                    pn.Column(cand_info, width=500),
                ),
                title="Detailed Analysis",
                styles={"background": "WhiteSmoke"},
                scroll=True,
            ),
        )

    def _record_user_action(self, action: str, data: Dict) -> None:
        if self.redo_stack:
            self.redo_stack = []
        self.undo_stack.append((action, data))

    def _undo_user_action(self) -> None:
        if len(self.undo_stack) == 0:
            return
        action, data = self.undo_stack.pop()
        recommendations = self.heatmap_recommendations

        if action == "discard":
            recommendations.append(data)
            self.redo_stack.append((action, data))
        else:
            for idx, d in enumerate(recommendations):
                candidate_name = d["source_column"]
                if candidate_name == data["source_column"]:
                    recommendations[idx] = data
                    self.redo_stack.append((action, d))
                    break
        self._write_json(recommendations)
        self._record_log("undo", data["source_column"], "")
        self._get_heatmap()
        return

    def _redo_user_action(self) -> None:
        if len(self.redo_stack) == 0:
            return
        action, data = self.redo_stack.pop()
        recommendations = self.heatmap_recommendations

        for idx, d in enumerate(recommendations):
            if d["source_column"] == data["source_column"]:
                if action == "discard":
                    recommendations.pop(idx)
                else:
                    recommendations[idx] = data
                self.undo_stack.append((action, d))
                break
        self._write_json(recommendations)
        self._record_log("redo", data["source_column"], "")
        self._get_heatmap()
        return

    def _record_log(self, action: str, source_column: str, target_column: str) -> None:
        timestamp = datetime.now()
        self.logs.append((timestamp, action, source_column, target_column))

    def _plot_history(self) -> pn.widgets.Tabulator:
        history_dict = {
            "Timestamp": [],
            "Action": [],
            "Source Column": [],
            "Target Column": [],
        }
        for timestamp, action, source_column, target_column in self.logs:
            history_dict["Timestamp"].append(timestamp)
            history_dict["Action"].append(action)
            if action in ["accept", "reject"]:
                history_dict["Source Column"].append(source_column)
                history_dict["Target Column"].append(target_column)

            elif action in ["undo", "redo", "discard"]:
                history_dict["Source Column"].append(source_column)
                history_dict["Target Column"].append("")

        history_df = pd.DataFrame(history_dict)

        return pn.widgets.Tabulator(history_df, show_index=False)

    def _plot_chat_pane(self) -> pn.chat.ChatInterface:
        def callback(contents: str, user: any, instance: any) -> Optional[str]:
            message = self.assistant.ask(contents)
            self.chat_history.append(message)
            return message

        return pn.chat.ChatInterface(callback=callback, scroll=True)

    def plot_heatmap(self) -> pn.Column:
        select_column = pn.widgets.Select(
            name="Column",
            options=list(self.source.columns),
            width=120,
        )

        select_candidate_type = pn.widgets.Select(
            name="Candidate type",
            options=["All", "enum", "number", "string", "boolean"],
            width=120,
        )

        n_similar_slider = pn.widgets.IntSlider(
            name="N Similar", start=0, end=5, value=0, width=100
        )
        thresh_slider = pn.widgets.FloatSlider(
            name="Threshold", start=0, end=1.0, step=0.01, value=0.1, width=100
        )

        acc_button = pn.widgets.Button(name="Accept Match", button_type="success")

        rej_button = pn.widgets.Button(name="Reject Match", button_type="danger")

        discard_button = pn.widgets.Button(name="Discard Column", button_type="warning")

        undo_button = pn.widgets.Button(
            name="Undo", button_style="outline", button_type="warning"
        )
        redo_button = pn.widgets.Button(
            name="Redo", button_style="outline", button_type="primary"
        )

        # Subschemas
        if not isinstance(self.target, pd.DataFrame) and self.target == "gdc":
            select_rec_groups = pn.widgets.MultiChoice(
                name="Recommendation subschema", options=self.subschemas, width=180
            )
            subschema_col = pn.Column(
                select_rec_groups,
            )

        def on_click_accept_match(event: Any) -> None:
            self._accept_match()

        def on_click_reject_match(event: Any) -> None:
            self._reject_match()

        def on_click_discard_column(event: Any) -> None:
            self._discard_column(select_column.value)

        def on_click_undo(event: Any) -> None:
            self._undo_user_action()

        def on_click_redo(event: Any) -> None:
            self._redo_user_action()

        acc_button.on_click(on_click_accept_match)
        rej_button.on_click(on_click_reject_match)
        discard_button.on_click(on_click_discard_column)
        undo_button.on_click(on_click_undo)
        redo_button.on_click(on_click_redo)

        heatmap_bind = pn.bind(
            self._plot_pane,
            select_column,
            select_candidate_type,
            (
                select_rec_groups
                if (not isinstance(self.target, pd.DataFrame) and self.target == "gdc")
                else None
            ),
            n_similar_slider,
            thresh_slider,
            acc_button.param.clicks,
            rej_button.param.clicks,
            discard_button.param.clicks,
            undo_button.param.clicks,
            redo_button.param.clicks,
        )

        buttons_down = pn.Column(acc_button, rej_button, discard_button)
        buttons_redo_undo = pn.Column(undo_button, redo_button)

        column_top = pn.Row(
            select_column,
            select_candidate_type,
            (
                subschema_col
                if (not isinstance(self.target, pd.DataFrame) and self.target == "gdc")
                else None
            ),
            n_similar_slider,
            thresh_slider,
            buttons_down,
            buttons_redo_undo,
            width=1200,
            styles=dict(background="WhiteSmoke"),
        )

        # AI Assistant
        if self.assistant:
            chat_pane = self._plot_chat_pane()

        return pn.Column(
            column_top,
            (
                pn.FloatPanel(
                    chat_pane,
                    name="Chat with AI Assistant",
                    width=600,
                    align="end",
                )
                if self.assistant
                else None
            ),
            pn.Spacer(height=5),
            pn.Column(heatmap_bind),
            scroll=True,
        )

    # For caching purposes
    def _get_data_checksum(self) -> str:
        return hashlib.sha1(pd.util.hash_pandas_object(self.source).values).hexdigest()

    def _get_ground_truth_checksum(self) -> str:
        if isinstance(self.target, pd.DataFrame):
            gt_checksum = hashlib.sha1(
                pd.util.hash_pandas_object(self.target).values
            ).hexdigest()
        else:
            gt_checksum = self.target
        return gt_checksum

    def _cache_results(self, reducings: List[Dict]) -> None:
        cache_path = join(
            BDIKIT_CACHE_DIR,
            f"reducings_{self._get_ground_truth_checksum()}_{self._get_data_checksum()}_{self.top_k}.json",
        )
        if not exists(cache_path):
            with open(cache_path, "w") as f:
                json.dump(reducings, f)

    def _load_cached_results(self) -> Optional[List[Dict]]:
        cache_path = join(
            BDIKIT_CACHE_DIR,
            f"reducings_{self._get_ground_truth_checksum()}_{self._get_data_checksum()}_{self.top_k}.json",
        )
        if exists(cache_path):
            with open(cache_path) as f:
                return json.load(f)
        return None

    def get_recommendations(
        self, source: pd.DataFrame, target: pd.DataFrame, top_k: int
    ) -> List[TopkMatching]:
        recommendations = []
        for reducings in self.heatmap_recommendations:
            recommendations.append(
                {
                    "source_column": reducings["source_column"],
                    "top_k_columns": [
                        ColumnScore(column_name=column[0], score=column[1])
                        for column in reducings["top_k_columns"]
                    ],
                }
            )
        return recommendations


class LLMAssistant:
    def __init__(self):
        self.client = OpenAI()

    def ask(self, context: str, model: str = "gpt-4-turbo-preview") -> Optional[str]:
        messages = [
            {
                "role": "system",
                "content": "You are an assistant for BDIViz Tool. A visualization tool for biomedical schema matching tasks. You are a helper for Expert-In-The-Loop biomedical researchers to provide discriptions for the columns in the dataset.",
            },
            {
                "role": "user",
                "content": context,
            },
        ]
        answers = self.client.chat.completions.create(
            model=model, messages=messages, temperature=0.3
        )
        return answers.choices[0].message.content

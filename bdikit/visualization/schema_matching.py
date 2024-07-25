import hashlib
import json
import logging
from os.path import dirname, exists, join
from typing import Any, Dict, List, Optional, Tuple, Union

import altair as alt
import numpy as np
import pandas as pd
import panel as pn
from Levenshtein import distance
from natsort import index_natsorted
from sklearn.cluster import AffinityPropagation

from bdikit import ValueMatchingResult, match_values
from bdikit.download import BDIKIT_CACHE_DIR
from bdikit.mapping_algorithms.column_mapping.topk_matchers import (
    CLTopkColumnMatcher,
    ColumnScore,
    TopkColumnMatcher,
    TopkMatching,
)
from bdikit.models.contrastive_learning.cl_api import (
    DEFAULT_CL_MODEL,
)
from bdikit.utils import get_gdc_layered_metadata, read_gdc_schema

GDC_DATA_PATH = join(dirname(__file__), "../resource/gdc_table.csv")
logger = logging.getLogger(__name__)

pn.extension("tabulator")
pn.extension("mathjax")
pn.extension("vega")


def generate_top_k_matches(
    source: pd.DataFrame, target: Union[pd.DataFrame, str], top_k: int = 10
) -> List[Dict]:
    if isinstance(target, pd.DataFrame):
        target_df = target
    elif target == "gdc":
        target_df = pd.read_csv(GDC_DATA_PATH)
    else:
        raise ValueError("Invalid target value. Must be a DataFrame or 'gdc'.")

    topk_matcher = CLTopkColumnMatcher(model_name=DEFAULT_CL_MODEL)
    top_k_matches = topk_matcher.get_recommendations(
        source, target=target_df, top_k=top_k
    )

    output_json = []
    for match in top_k_matches:
        source_dict = {
            "source_column": match["source_column"],
            "top_k_columns": [],
        }
        for column in match["top_k_columns"]:
            source_dict["top_k_columns"].append(
                [column.column_name, float(column.score)]
            )
        output_json.append(source_dict)
    return output_json


def generate_value_matches(
    source: pd.DataFrame,
    target: Union[pd.DataFrame, str],
    column_mapping: Tuple[str, str],
) -> List[ValueMatchingResult]:
    if not isinstance(target, pd.DataFrame) and target == "gdc":
        target = pd.read_csv(GDC_DATA_PATH)
    mapping_df = pd.DataFrame(
        [
            {
                "source": column_mapping[0],
                "target": column_mapping[1],
            }
        ]
    )
    mappings = match_values(
        source,
        column_mapping=mapping_df,
        target=target,
        method="tfidf",
    )
    return mappings


def gdc_clean_heatmap_recommendations(
    heatmap_recommendations: List[Dict], max_chars_samples: int = 150
) -> Dict[str, pd.DataFrame]:
    gdc_metadata = get_gdc_layered_metadata()

    candidates_dfs = {}

    for column_data in heatmap_recommendations:
        column_name = column_data["source_column"]
        recommendations = []
        for candidate_name, candidate_similarity in column_data["top_k_columns"]:
            subschema, gdc_data = gdc_metadata[candidate_name]
            candidate_description = gdc_data.get("description", "")
            candidate_description = candidate_description
            candidate_values = ", ".join(gdc_data.get("enum", []))
            candidate_values = truncate_text(candidate_values, max_chars_samples)
            recommendations.append(
                (
                    candidate_name,
                    candidate_similarity,
                    candidate_description,
                    candidate_values,
                    subschema,
                )
            )

        candidates_dfs[column_name] = pd.DataFrame(
            recommendations,
            columns=[
                "Candidate",
                "Similarity",
                "Description",
                "Values (sample)",
                "Subschema",
            ],
        )

    return candidates_dfs


def truncate_text(text: str, max_chars: int):
    if len(text) > max_chars:
        return text[:max_chars] + "..."
    else:
        return text


class BDISchemaMatchingHeatMap(TopkColumnMatcher):
    def __init__(
        self,
        source: pd.DataFrame,
        target: Union[pd.DataFrame, str] = "gdc",
        top_k: int = 10,
        heatmap_recommendations: Optional[List[Dict]] = None,
        max_chars_samples: int = 150,
        height: int = 600,
    ) -> None:
        self.json_path = "heatmap_recommendations.json"
        self.source = source
        self.target = target  # IMPORTANT!!!
        self.top_k = max(1, min(top_k, 20))

        self.rec_table_df = None
        self.rec_list_df = None
        self.rec_cols = None
        self.subschemas = None
        self.rec_cols_gdc = None
        self.clusters = None

        self.gdc_metadata = get_gdc_layered_metadata()

        # Selected column
        self.selected_row = None

        # Load cached results
        cached_heatmap_recommendations = self._load_cached_results()
        if cached_heatmap_recommendations is not None:
            self.heatmap_recommendations = cached_heatmap_recommendations
        else:
            if heatmap_recommendations is None:
                self.heatmap_recommendations = generate_top_k_matches(
                    self.source, target=target, top_k=self.top_k
                )
            else:
                self.heatmap_recommendations = heatmap_recommendations
            self._cache_results(self.heatmap_recommendations)

        self._write_json(self.heatmap_recommendations)

        if not isinstance(target, pd.DataFrame) and target == "gdc":
            self.candidates_dfs = gdc_clean_heatmap_recommendations(
                self.heatmap_recommendations, max_chars_samples=max_chars_samples
            )
        self.height = height

        # Undo/Redo
        # The undo/redo stack is a list of data
        # Data is like this: {'Candidate column': 'Country', 'Top k columns': [['country_of_birth', '0.5726'], ...]}
        self.undo_stack = []
        self.redo_stack = []

    def _load_json(self) -> List[Dict]:
        with open(self.json_path) as f:
            data = json.load(f)
        return data

    def _write_json(self, data: List[Dict]) -> None:
        self.heatmap_recommendations = data
        with open(self.json_path, "w") as f:
            json.dump(data, f)

    def get_heatmap(self) -> None:
        recommendations = self._load_json()
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
                }
                # [GDC] get description and values
                if not isinstance(self.target, pd.DataFrame) and self.target == "gdc":
                    candidates_info = self.candidates_dfs[d["source_column"]]
                    cadidate_info = candidates_info[
                        candidates_info["Candidate"] == c[0]
                    ]
                    rec_row["Description"] = cadidate_info["Description"].values[0]
                    rec_row["Values (sample)"] = cadidate_info[
                        "Values (sample)"
                    ].values[0]
                    rec_row["Subschema"] = cadidate_info["Subschema"].values[0]
                rec_list.append(rec_row)
            rec_table.append(col_dict)

        rec_cols = list(rec_cols)
        rec_cols.sort()

        rec_table_df = pd.DataFrame(rec_table)
        rec_list_df = pd.DataFrame(rec_list)
        rec_list_df["Value"] = pd.to_numeric(rec_list_df["Value"])

        self.rec_table_df = rec_table_df
        self.rec_list_df = rec_list_df
        self.rec_cols = rec_cols

        # [GDC] get subschema information
        if not isinstance(self.target, pd.DataFrame) and self.target == "gdc":
            self.get_cols_subschema()

        self.get_clusters()

    def get_cols_subschema(self) -> None:
        subschemas = []
        schema = read_gdc_schema()
        data_dict = {
            "parent": [],
            "column_name": [],
            "column_type": [],
            "column_description": [],
            "column_values": [],
        }
        for parent, values in schema.items():
            for candidate in values["properties"].keys():
                if candidate in self.rec_cols:
                    if parent not in subschemas:
                        subschemas.append(parent)
                    data_dict["parent"].append(parent)
                    data_dict["column_name"].append(candidate)
                    data_dict["column_type"].append(
                        self._gdc_get_column_type(values["properties"][candidate])
                    )
                    data_dict["column_description"].append(
                        self._gdc_get_column_description(
                            values["properties"][candidate]
                        )
                    )
                    data_dict["column_values"].append(
                        self._gdc_get_column_values(values["properties"][candidate])
                    )

        self.subschemas = subschemas
        self.rec_cols_gdc = pd.DataFrame(data_dict)

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

    def _accept_match(self) -> None:
        if self.selected_row is None:
            return
        col_name = self.selected_row["Column"].values[0]
        match_name = self.selected_row["Recommendation"].values[0]
        recommendations = self._load_json()
        for idx, d in enumerate(recommendations):
            candidate_name = d["source_column"]
            if candidate_name != col_name:
                continue
            for top_k_name, top_k_score in d["top_k_columns"]:
                if top_k_name == match_name:
                    recommendations[idx] = {
                        "source_column": candidate_name,
                        "top_k_columns": [[top_k_name, top_k_score]],
                    }

                    # record the action
                    self._record_user_action("accept", d)

                    self._write_json(recommendations)
                    self.get_heatmap()
                    return

    def _reject_match(self) -> None:
        if self.selected_row is None:
            return
        col_name = self.selected_row["Column"].values[0]
        match_name = self.selected_row["Recommendation"].values[0]
        recommendations = self._load_json()
        for idx, d in enumerate(recommendations):
            candidate_name = d["source_column"]
            if candidate_name != col_name:
                continue
            new_top_k = []
            for top_k_name, top_k_score in d["top_k_columns"]:
                if top_k_name != match_name:
                    new_top_k.append([top_k_name, top_k_score])
            recommendations[idx] = {
                "source_column": candidate_name,
                "top_k_columns": new_top_k,
            }

            # record the action
            self._record_user_action("reject", d)

            self._write_json(recommendations)
            self.get_heatmap()
            return

    def get_clusters(self) -> None:
        words = self.rec_table_df["Column"].to_numpy()
        lev_similarity = -1 * np.array(
            [[distance(w1, w2) for w1 in words] for w2 in words]
        )
        lev_similarity = lev_similarity.astype(np.float32)

        affprop = AffinityPropagation(
            affinity="precomputed", max_iter=1000, damping=0.7
        )
        affprop.fit(lev_similarity)

        logger.debug(f"Number of clusters: {np.unique(affprop.labels_).shape[0]}\n")
        cluster_names = []
        clusters = {}
        for cluster_id in np.unique(affprop.labels_):
            exemplar = words[affprop.cluster_centers_indices_[cluster_id]]
            cluster = np.unique(words[np.nonzero(affprop.labels_ == cluster_id)])
            cluster_str = ", ".join(cluster)
            logger.debug(" - *%s:* %s" % (exemplar, cluster_str))
            cluster_names.append(exemplar)
            clusters[exemplar] = cluster
        self.clusters = clusters

    def _plot_heatmap_base(
        self, heatmap_rec_list: pd.DataFrame, show_subschema: bool
    ) -> pn.pane.Vega:
        single = alt.selection_point(name="single")

        tooltip = [
            alt.Tooltip("Column", title="Column"),
            alt.Tooltip("Recommendation", title="Recommendation"),
            alt.Tooltip("Value", title="Similarity"),
        ]
        facet = alt.Facet(alt.Undefined)
        if not isinstance(self.target, pd.DataFrame) and self.target == "gdc":
            tooltip.extend(
                [
                    alt.Tooltip("Description", title="Description"),
                    alt.Tooltip("Values (sample)", title="Values (sample)"),
                ]
            )
            if show_subschema:
                facet = alt.Facet("Subschema:O", columns=1)

        base = (
            alt.Chart(heatmap_rec_list)
            .mark_rect(size=100)
            .encode(
                y=alt.Y("Column:O", sort=None),
                x=alt.X(f"Recommendation:O", sort=None).axis(labelAngle=-45),
                color=alt.condition(
                    single,
                    alt.Color("Value:Q").scale(domainMax=1, domainMin=0),
                    alt.value("lightgray"),
                ),
                # color="Value:Q",
                tooltip=tooltip,
                facet=facet,
            )
            .add_params(single)
            .configure(background="#f5f5f5")
        )
        return pn.pane.Vega(base)

    def _update_column_selection(
        self, heatmap_rec_list: pd.DataFrame, selection: List[int]
    ) -> Tuple[str, str]:
        selected_idx = [selection[0] - 1]
        selected_row = heatmap_rec_list.iloc[selected_idx]
        self.selected_row = selected_row
        column = selected_row["Column"].values[0]
        rec = selected_row["Recommendation"].values[0]
        return column, rec

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
        column, rec = self._update_column_selection(heatmap_rec_list, selection)
        if not isinstance(self.target, pd.DataFrame) and self.target == "gdc":
            df = self.candidates_dfs[column][
                self.candidates_dfs[column]["Candidate"] == rec
            ]

            rec_rank = df.index[0]
            _, gdc_data = self.gdc_metadata[rec]
            values = gdc_data.get("enum", [])

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
        column, rec = self._update_column_selection(heatmap_rec_list, selection)
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
    ) -> "pn.pane.Markdown | alt.Chart":
        if pd.api.types.is_numeric_dtype(dataset[column]):
            x = alt.X(column, bin=True).axis(labelAngle=-45)
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
                x = alt.X(
                    column + ":N",
                    sort=values,
                ).axis(labelAngle=-45)

        chart = (
            alt.Chart(dataset.fillna("Null"), height=300)
            .mark_bar()
            .encode(
                x=x,
                y="count()",
            )
            .properties(width="container", title="Histogram of " + column)
            .configure(background="#f5f5f5")
        )
        return chart

    def _plot_target_histogram(
        self, heatmap_rec_list: pd.DataFrame, selection: List[int]
    ) -> "pn.pane.Markdown | alt.Chart":
        if not isinstance(self.target, pd.DataFrame):
            return pn.pane.Markdown("No ground truth provided.")
        if not selection:
            return pn.pane.Markdown("## No selection")

        _, rec = self._update_column_selection(heatmap_rec_list, selection)

        return self._plot_column_histogram(rec, self.target)

    def _plot_value_matches(
        self, heatmap_rec_list: pd.DataFrame, selection: List[int]
    ) -> "pn.widgets.DataFrame | pn.pane.Markdown":
        if not selection:
            return pn.pane.Markdown("## No selection")

        column, rec = self._update_column_selection(heatmap_rec_list, selection)

        column_mapping = (column, rec)
        mappings = generate_value_matches(self.source, self.target, column_mapping)
        if not mappings or not mappings[0]["matches"]:
            return pn.pane.Markdown("No value matches found.")
        mapping = mappings[0]
        df = pd.DataFrame(
            {
                "Source Value": [match.source_value for match in mapping["matches"]],
                "Target Value": [match.target_value for match in mapping["matches"]],
                "Similarity": [match.similarity for match in mapping["matches"]],
            }
        )

        return pn.widgets.DataFrame(
            df,
            name=f"Value Matches {mapping['source']} to {mapping['target']}",
            height=200,
        )

    def _plot_pane(
        self,
        select_column: Optional[str] = None,
        subschemas: List[str] = [],
        n_similar: int = 0,
        threshold: float = 0.5,
        show_subschema: bool = False,
        acc_click: int = 0,
        rej_click: int = 0,
        undo_click: int = 0,
        redo_click: int = 0,
    ) -> pn.Column:
        heatmap_rec_list = self.rec_list_df[self.rec_list_df["Value"] >= threshold]
        if select_column:
            clustered_cols = []
            for cluster_key, cluster_list in self.clusters.items():
                cluster_list = cluster_list.tolist()
                if select_column in cluster_list:
                    clustered_cols.extend(cluster_list)
                    similarities = [distance(w1, select_column) for w1 in cluster_list]
                    clustered_cols = sorted(
                        cluster_list, key=lambda x: similarities[cluster_list.index(x)]
                    )
                    clustered_cols = clustered_cols[: n_similar + 1]
            heatmap_rec_list = heatmap_rec_list[
                heatmap_rec_list["Column"].isin(clustered_cols)
            ]
            heatmap_rec_list = heatmap_rec_list.sort_values(
                by="Column",
                key=lambda x: np.argsort(
                    index_natsorted(
                        heatmap_rec_list["Column"].apply(
                            lambda x: clustered_cols.index(x)
                        )
                    )
                ),
            )

        if subschemas:
            subschema_rec_cols = self.rec_cols_gdc[
                self.rec_cols_gdc["parent"].isin(subschemas)
            ]["column_name"].to_list()
            heatmap_rec_list = heatmap_rec_list[
                heatmap_rec_list["Recommendation"].isin(subschema_rec_cols)
            ]

        heatmap_pane = self._plot_heatmap_base(heatmap_rec_list, show_subschema)

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

        column_hist = self._plot_column_histogram(select_column, self.source)

        value_matches = pn.bind(
            self._plot_value_matches,
            heatmap_rec_list,
            heatmap_pane.selection.param.single,
        )

        return pn.Column(
            pn.Row(
                heatmap_pane,
                scroll=True,
                width=1200,
                styles=dict(background="WhiteSmoke"),
            ),
            pn.Spacer(height=5),
            pn.Card(
                value_matches,
                title="Value Matches",
                styles={"background": "WhiteSmoke"},
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
        self.undo_stack.append(data)

    def _undo_user_action(self) -> None:
        if len(self.undo_stack) == 0:
            return
        data = self.undo_stack.pop()
        recommendations = self._load_json()
        for idx, d in enumerate(recommendations):
            candidate_name = d["source_column"]
            if candidate_name == data["source_column"]:
                recommendations[idx] = data

                self.redo_stack.append(d)
                self._write_json(recommendations)
                self.get_heatmap()
                return

    def _redo_user_action(self) -> None:
        if len(self.redo_stack) == 0:
            return
        data = self.redo_stack.pop()
        recommendations = self._load_json()
        for idx, d in enumerate(recommendations):
            candidate_name = d["source_column"]
            if candidate_name == data["source_column"]:
                recommendations[idx] = data

                self.undo_stack.append(d)
                self._write_json(recommendations)
                self.get_heatmap()
                return

    def plot_heatmap(self) -> pn.Column:
        select_column = pn.widgets.Select(
            name="Column",
            options=list(self.rec_table_df["Column"]),
            width=120,
        )
        # select_cluster = pn.widgets.MultiChoice(
        #     name="Column cluster", options=list(self.clusters.keys()), width=220
        # )

        n_similar_slider = pn.widgets.IntSlider(
            name="N Similar", start=0, end=5, value=0, width=100
        )
        thresh_slider = pn.widgets.FloatSlider(
            name="Threshold", start=0, end=1.0, step=0.01, value=0.1, width=100
        )

        acc_button = pn.widgets.Button(name="Accept Match", button_type="success")

        rej_button = pn.widgets.Button(name="Decline Match", button_type="danger")

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
            show_subschema = pn.widgets.Checkbox(name="Show subschema", value=False)
            subschema_col = pn.Column(
                select_rec_groups,
                show_subschema,
            )

        def on_click_accept_match(event: Any) -> None:
            self._accept_match()

        def on_click_reject_match(event: Any) -> None:
            self._reject_match()

        def on_click_undo(event: Any) -> None:
            self._undo_user_action()

        def on_click_redo(event: Any) -> None:
            self._redo_user_action()

        acc_button.on_click(on_click_accept_match)
        rej_button.on_click(on_click_reject_match)
        undo_button.on_click(on_click_undo)
        redo_button.on_click(on_click_redo)

        heatmap_bind = pn.bind(
            self._plot_pane,
            select_column,
            (
                select_rec_groups
                if (not isinstance(self.target, pd.DataFrame) and self.target == "gdc")
                else None
            ),
            n_similar_slider,
            thresh_slider,
            (
                show_subschema
                if (not isinstance(self.target, pd.DataFrame) and self.target == "gdc")
                else False
            ),
            acc_button.param.clicks,
            rej_button.param.clicks,
            undo_button.param.clicks,
            redo_button.param.clicks,
        )

        buttons_down = pn.Column(acc_button, rej_button)
        buttons_redo_undo = pn.Column(undo_button, redo_button)

        column_top = pn.Row(
            select_column,
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

        return pn.Column(
            column_top, pn.Spacer(height=5), pn.Column(heatmap_bind), scroll=True
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

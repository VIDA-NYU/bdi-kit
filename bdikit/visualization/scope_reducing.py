import json
import logging
import altair as alt
import numpy as np
import pandas as pd
import panel as pn
from bdikit.utils import read_gdc_schema
from bdikit.utils import get_gdc_metadata
from Levenshtein import distance
from sklearn.cluster import AffinityPropagation

logger = logging.getLogger(__name__)

pn.extension("tabulator")
pn.extension("mathjax")
pn.extension("vega")


def clean_reduced_scope(reduced_scope, max_chars_samples):
    gdc_metadata = get_gdc_metadata()

    candidates_dfs = {}

    for column_data in reduced_scope:
        column_name = column_data["Candidate column"]
        recommendations = []
        for candidate_name, candidate_similarity in column_data["Top k columns"]:
            candidate_description = gdc_metadata[candidate_name].get("description", "")
            candidate_description = candidate_description
            candidate_values = ", ".join(gdc_metadata[candidate_name].get("enum", []))
            candidate_values = truncate_text(candidate_values, max_chars_samples)
            recommendations.append(
                (
                    candidate_name,
                    candidate_similarity,
                    candidate_description,
                    candidate_values,
                )
            )

        candidates_dfs[column_name] = pd.DataFrame(
            recommendations,
            columns=["Candidate", "Similarity", "Description", "Values (sample)"],
        )

    return candidates_dfs


def truncate_text(text, max_chars):
    if len(text) > max_chars:
        return text[:max_chars] + "..."
    else:
        return text


class ScopeReducerExplorer:
    def __init__(
        self, dataset, reduced_scope, max_chars_samples=150, height=600
    ) -> None:
        self.dataset = dataset
        self.reduced_scope = reduced_scope
        self.candidates_dfs = clean_reduced_scope(
            reduced_scope, max_chars_samples=max_chars_samples
        )
        self.height = height
        self.max_candidates = len(reduced_scope[0]["Top k columns"])

    def _plot_column_histogram(self, column):
        if self.dataset[column].dtype == "float64":
            print(column)
            chart = (
                alt.Chart(self.dataset.fillna("Null"), height=300)
                .mark_bar()
                .encode(
                    alt.X(column, bin=True),
                    y="count()",
                )
                .properties(width="container", title="Histogram of " + column)
            )
            return chart
        else:
            values = list(self.dataset[column].unique())
            if len(values) == len(self.dataset[column]):
                string = f"""Values are unique. 
                Some samples: {values[:5]}"""
                return pn.pane.Markdown(string)
            else:
                if np.nan in values:
                    values.remove(np.nan)
                values.sort()

                chart = (
                    alt.Chart(self.dataset.fillna("Null"), height=300)
                    .mark_bar()
                    .encode(
                        x=alt.X(
                            column + ":N",
                            sort=values,
                        ),
                        y="count()",
                    )
                    .properties(width="container", title="Histogram of " + column)
                )
        return chart

    def _candidates_table(self, column, n_candidates):
        df = self.candidates_dfs[column].loc[: n_candidates - 1]

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
            height=self.height,
            embed_content=True,
            header_align="center",
            theme="simple",
            disabled=True,
        )
        return table_candidates

    def explore(self):
        select_column = pn.widgets.Select(
            name="Column selected",
            options=list(self.candidates_dfs.keys()),
            align="center",
        )
        select_n_candidates = pn.widgets.EditableIntSlider(
            name="Number of candidates",
            start=1,
            end=self.max_candidates,
            step=1,
            value=min(5, self.max_candidates),
            align="center",
        )
        cand_table = pn.bind(self._candidates_table, select_column, select_n_candidates)
        column_hist = pn.bind(self._plot_column_histogram, select_column)

        explorer = pn.Column(
            pn.Row(
                "# Scope Reducing Explorer",
                pn.Spacer(width=30),
                select_column,
                pn.Spacer(width=30),
                select_n_candidates,
                align=("start", "center"),
            ),
            pn.Spacer(height=30),
            pn.Row(pn.Column(column_hist, width=500), pn.Spacer(width=30), cand_table),
            styles=dict(background="white"),
        )

        return explorer


class SRHeatMapManager:
    def __init__(self) -> None:
        self.json_path = "reduced_scope.json"

        self.rec_table_df = None
        self.rec_list_df = None
        self.rec_cols = None
        self.subschemas = None
        self.rec_cols_gdc = None
        self.clusters = None

        # Selected column
        self.selected_row = None

    def _load_json(self):
        with open(self.json_path) as f:
            data = json.load(f)
        return data

    def _write_json(self, data):
        with open(self.json_path, "w") as f:
            json.dump(data, f)

    def get_heatmap(self):
        recommendations = self._load_json()
        rec_cols = set()
        rec_table = []
        rec_list = []

        for d in recommendations:
            col_dict = {"Column": d["Candidate column"]}
            for c in d["Top k columns"]:
                rec_cols.add(c[0])
                col_dict[c[0]] = c[1]
                rec_list.append(
                    {
                        "Column": d["Candidate column"],
                        "Recommendation": c[0],
                        "Value": c[1],
                    }
                )
            rec_table.append(col_dict)

        rec_cols = list(rec_cols)
        rec_cols.sort()

        rec_table_df = pd.DataFrame(rec_table)
        rec_list_df = pd.DataFrame(rec_list)
        rec_list_df["Value"] = pd.to_numeric(rec_list_df["Value"])

        self.rec_table_df = rec_table_df
        self.rec_list_df = rec_list_df
        self.rec_cols = rec_cols

        self.get_cols_subschema()
        self.get_clusters()

    def get_cols_subschema(self):
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
                        self._get_column_type(values["properties"][candidate])
                    )
                    data_dict["column_description"].append(
                        self._get_column_description(values["properties"][candidate])
                    )
                    data_dict["column_values"].append(
                        self._get_column_values(values["properties"][candidate])
                    )

        self.subschemas = subschemas
        self.rec_cols_gdc = pd.DataFrame(data_dict)

    def _get_column_type(self, properties):
        if "enum" in properties:
            return "enum"
        elif "type" in properties:
            return properties["type"]
        else:
            return None

    def _get_column_description(self, properties):
        if "description" in properties:
            return properties["description"]
        elif "common" in properties:
            return properties["common"]["description"]
        return ""

    def _get_column_values(self, properties):
        col_type = self._get_column_type(properties)
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

    def _accept_match(self, col_name=None, match_name=None):
        if self.selected_row is None:
            return
        col_name = self.selected_row["Column"].values[0]
        match_name = self.selected_row["Recommendation"].values[0]
        recommendations = self._load_json()
        for idx, d in enumerate(recommendations):
            candidate_name = d["Candidate column"]
            if candidate_name != col_name:
                continue
            for top_k_name, top_k_score in d["Top k columns"]:
                if top_k_name == match_name:
                    recommendations[idx] = {
                        "Candidate column": candidate_name,
                        "Top k columns": [[top_k_name, top_k_score]],
                    }
                    self._write_json(recommendations)
                    self.get_heatmap()
                    return

    def _reject_match(self):
        if self.selected_row is None:
            return
        col_name = self.selected_row["Column"].values[0]
        match_name = self.selected_row["Recommendation"].values[0]
        recommendations = self._load_json()
        for idx, d in enumerate(recommendations):
            candidate_name = d["Candidate column"]
            if candidate_name != col_name:
                continue
            new_top_k = []
            for top_k_name, top_k_score in d["Top k columns"]:
                if top_k_name != match_name:
                    new_top_k.append([top_k_name, top_k_score])
            recommendations[idx] = {
                "Candidate column": candidate_name,
                "Top k columns": new_top_k,
            }
            self._write_json(recommendations)
            self.get_heatmap()
            return

    def get_clusters(self):
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

    def _plot_heatmap_base(self, heatmap_rec_list):
        single = alt.selection_point(name="single")
        base = (
            alt.Chart(heatmap_rec_list)
            .mark_rect(size=100)
            .encode(
                y=alt.X("Column:O", sort=None),
                x=alt.X(f"Recommendation:O", sort=None),
                color=alt.condition(single, "Value:Q", alt.value("lightgray")),
                # color="Value:Q",
                tooltip=[
                    alt.Tooltip("Column", title="Column"),
                    alt.Tooltip("Recommendation", title="Recommendation"),
                    alt.Tooltip("Value", title="Value"),
                ],
            )
            .add_params(single)
        )
        return pn.pane.Vega(base)

    def _plot_selected_row(self, heatmap_rec_list, selection):
        if not selection:
            return "## No selection"
        selected_row = heatmap_rec_list.iloc[selection]
        column = selected_row["Column"].values[0]
        rec = selected_row["Recommendation"].values[0]
        # value = selected_row["Value"]
        # self._accept_match(column, rec)
        self.selected_row = selected_row
        return pn.widgets.DataFrame(selected_row)

    def _plot_pane(
        self, clusters=[], subschemas=[], threshold=0.5, acc_click=0, rej_click=0
    ):
        heatmap_rec_list = self.rec_list_df[self.rec_list_df["Value"] >= threshold]
        if clusters:
            clustered_cols = []
            for cluster in clusters:
                clustered_cols.extend(self.clusters[cluster])
            heatmap_rec_list = heatmap_rec_list[
                heatmap_rec_list["Column"].isin(clustered_cols)
            ]
        if subschemas:
            subschema_rec_cols = self.rec_cols_gdc[
                self.rec_cols_gdc["parent"].isin(subschemas)
            ]["column_name"].to_list()
            heatmap_rec_list = heatmap_rec_list[
                heatmap_rec_list["Recommendation"].isin(subschema_rec_cols)
            ]

        heatmap_pane = self._plot_heatmap_base(heatmap_rec_list)
        return pn.Column(
            heatmap_pane,
            pn.bind(
                self._plot_selected_row,
                heatmap_rec_list,
                heatmap_pane.selection.param.single,
            ),
        )

    def plot_heatmap(self):
        select_cluster = pn.widgets.MultiChoice(
            name="Column cluster", options=list(self.clusters.keys()), width=220
        )
        select_rec_groups = pn.widgets.MultiChoice(
            name="Recommendation subschema", options=self.subschemas, width=220
        )
        thresh_slider = pn.widgets.EditableFloatSlider(
            name="Threshold", start=0, end=1.0, step=0.01, value=0.5, width=220
        )

        acc_button = pn.widgets.Button(name="Accept Match", button_type="success")

        rej_button = pn.widgets.Button(name="Decline Match", button_type="danger")

        def on_click_accept_match(event):
            self._accept_match()

        def on_click_reject_match(event):
            self._reject_match()

        acc_button.on_click(on_click_accept_match)
        rej_button.on_click(on_click_reject_match)

        heatmap_bind = pn.bind(
            self._plot_pane,
            select_cluster,
            select_rec_groups,
            thresh_slider,
            acc_button.param.clicks,
            rej_button.param.clicks,
        )

        column_left = pn.Column(
            "# Column",
            select_cluster,
            select_rec_groups,
            thresh_slider,
            acc_button,
            rej_button,
            styles=dict(background="WhiteSmoke"),
        )

        return pn.Row(column_left, pn.Column(heatmap_bind), scroll=True)

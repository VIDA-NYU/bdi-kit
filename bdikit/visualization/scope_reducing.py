import json
import logging

import altair as alt
import numpy as np
import pandas as pd
import panel as pn
from bdikit.utils import get_gdc_layered_metadata, read_gdc_schema
from Levenshtein import distance
from natsort import index_natsorted
from sklearn.cluster import AffinityPropagation

logger = logging.getLogger(__name__)

pn.extension("tabulator")
pn.extension("mathjax")
pn.extension("vega")


def gdc_clean_reduced_scope(reduced_scope, max_chars_samples):
    gdc_metadata = get_gdc_layered_metadata()

    candidates_dfs = {}

    for column_data in reduced_scope:
        column_name = column_data["Candidate column"]
        recommendations = []
        for candidate_name, candidate_similarity in column_data["Top k columns"]:
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


def truncate_text(text, max_chars):
    if len(text) > max_chars:
        return text[:max_chars] + "..."
    else:
        return text


class SRHeatMapManager:
    def __init__(
        self,
        dataset,
        reduced_scope=None,
        max_chars_samples=150,
        height=600,
        ground_truth="gdc",
    ) -> None:
        self.json_path = "reduced_scope.json"
        self.dataset = dataset
        self.ground_truth = ground_truth  # IMPORTANT!!!

        self.rec_table_df = None
        self.rec_list_df = None
        self.rec_cols = None
        self.subschemas = None
        self.rec_cols_gdc = None
        self.clusters = None

        self.gdc_metadata = get_gdc_layered_metadata()

        # Selected column
        self.selected_row = None

        if reduced_scope is not None:
            self._write_json(reduced_scope)
        self.reduced_scope = self._load_json()

        if ground_truth == "gdc":
            self.candidates_dfs = gdc_clean_reduced_scope(
                self.reduced_scope, max_chars_samples=max_chars_samples
            )
        self.height = height

        # Undo/Redo
        # The undo/redo stack is a list of data
        # Data is like this: {'Candidate column': 'Country', 'Top k columns': [['country_of_birth', '0.5726'], ...]}
        self.undo_stack = []
        self.redo_stack = []

    def _load_json(self):
        with open(self.json_path) as f:
            data = json.load(f)
        return data

    def _write_json(self, data):
        self.reduced_scope = data
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
                rec_row = {
                    "Column": d["Candidate column"],
                    "Recommendation": c[0],
                    "Value": c[1],
                }
                # [GDC] get description and values
                if self.ground_truth == "gdc":
                    candidates_info = self.candidates_dfs[d["Candidate column"]]
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
        if self.ground_truth == "gdc":
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

    def _gdc_get_column_type(self, properties):
        if "enum" in properties:
            return "enum"
        elif "type" in properties:
            return properties["type"]
        else:
            return None

    def _gdc_get_column_description(self, properties):
        if "description" in properties:
            return properties["description"]
        elif "common" in properties:
            return properties["common"]["description"]
        return ""

    def _gdc_get_column_values(self, properties):
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

                    # record the action
                    self._record_user_action("accept", d)

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

            # record the action
            self._record_user_action("reject", d)
            
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

    def _plot_heatmap_base(self, heatmap_rec_list, show_subschema):
        single = alt.selection_point(name="single")
        if self.ground_truth == "gdc":
            if show_subschema:
                base = (
                    alt.Chart(heatmap_rec_list)
                    .mark_rect(size=100)
                    .encode(
                        y=alt.X("Column:O", sort=None),
                        x=alt.X(f"Recommendation:O", sort=None),
                        color=alt.condition(
                            single,
                            alt.Color("Value:Q").scale(domainMax=1, domainMin=0),
                            alt.value("lightgray"),
                        ),
                        # color="Value:Q",
                        tooltip=[
                            alt.Tooltip("Column", title="Column"),
                            alt.Tooltip("Recommendation", title="Recommendation"),
                            alt.Tooltip("Value", title="Similarity"),
                            alt.Tooltip("Description", title="Description"),
                            alt.Tooltip("Values (sample)", title="Values (sample)"),
                        ],
                        facet=alt.Facet("Subschema:O", columns=1),
                    )
                    .add_params(single)
                    .configure(background="#f5f5f5")
                )
            else:
                base = (
                    alt.Chart(heatmap_rec_list)
                    .mark_rect(size=100)
                    .encode(
                        y=alt.X("Column:O", sort=None),
                        x=alt.X(f"Recommendation:O", sort=None),
                        color=alt.condition(
                            single,
                            alt.Color("Value:Q").scale(domainMax=1, domainMin=0),
                            alt.value("lightgray"),
                        ),
                        # color="Value:Q",
                        tooltip=[
                            alt.Tooltip("Column", title="Column"),
                            alt.Tooltip("Recommendation", title="Recommendation"),
                            alt.Tooltip("Value", title="Similarity"),
                            alt.Tooltip("Description", title="Description"),
                            alt.Tooltip("Values (sample)", title="Values (sample)"),
                        ],
                    )
                    .add_params(single)
                    .configure(background="#f5f5f5")
                )
        else:
            base = (
                alt.Chart(heatmap_rec_list)
                .mark_rect(size=100)
                .encode(
                    y=alt.X("Column:O", sort=None),
                    x=alt.X(f"Recommendation:O", sort=None),
                    color=alt.condition(
                        single,
                        alt.Color("Value:Q").scale(domainMax=1, domainMin=0),
                        alt.value("lightgray"),
                    ),
                    # color="Value:Q",
                    tooltip=[
                        alt.Tooltip("Column", title="Column"),
                        alt.Tooltip("Recommendation", title="Recommendation"),
                    ],
                )
                .add_params(single)
                .configure(background="#f5f5f5")
            )
        return pn.pane.Vega(base)

    def _candidates_info(self, heatmap_rec_list, selection, n_samples=20):
        if not selection:
            return pn.pane.Markdown(
                """
                
                ### Selected Recommendation
                
                *No selection.*

            """
            )
        selection[0] -= 1
        selected_row = heatmap_rec_list.iloc[selection]
        self.selected_row = selected_row
        column = selected_row["Column"].values[0]
        rec = selected_row["Recommendation"].values[0]
        if self.ground_truth == "gdc":
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
        else:
            rec_info = f"""
                ### Selected Recommendation

                **Name:** {rec}

                **Similarity:** {selected_row["Value"].values[0]}

            """
        rec_pane = pn.pane.Markdown(rec_info)
        return rec_pane

    def _candidates_table(self, heatmap_rec_list, selection):
        if not selection:
            return pn.pane.Markdown("## No selection")
        selection[0] -= 1
        selected_row = heatmap_rec_list.iloc[selection]
        self.selected_row = selected_row
        column = selected_row["Column"].values[0]
        rec = selected_row["Recommendation"].values[0]
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

    def _plot_column_histogram(self, column):
        if self.dataset[column].dtype == "float64":
            chart = (
                alt.Chart(self.dataset.fillna("Null"), height=300)
                .mark_bar()
                .encode(
                    alt.X(column, bin=True),
                    y="count()",
                )
                .properties(width="container", title="Histogram of " + column)
                .configure(background="#f5f5f5")
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
                    .configure(background="#f5f5f5")
                )
        return chart

    def _plot_pane(
        self,
        select_column=None,
        subschemas=[],
        n_similar=0,
        threshold=0.5,
        show_subschema=False,
        acc_click=0,
        rej_click=0,
        undo_click=0,
        redo_click=0,
    ):
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
        cand_info = pn.bind(
            self._candidates_info,
            heatmap_rec_list,
            heatmap_pane.selection.param.single,
        )
        column_hist = self._plot_column_histogram(select_column)
        return pn.Column(
            pn.Row(
                heatmap_pane,
                scroll=True,
                width=1200,
                styles=dict(background="WhiteSmoke"),
            ),
            pn.Spacer(height=5),
            pn.Card(
                pn.Row(
                    pn.Column(column_hist, width=500),
                    pn.Column(cand_info, width=700),
                ),
                title="Detailed Analysis",
                styles={"background": "WhiteSmoke"},
                scroll=True,
            ),
        )

    def _record_user_action(self, action, data):
        if self.redo_stack:
            self.redo_stack = []
        self.undo_stack.append(data)
    
    def _undo_user_action(self):
        if len(self.undo_stack) == 0:
            return
        data = self.undo_stack.pop()
        recommendations = self._load_json()
        for idx, d in enumerate(recommendations):
            candidate_name = d["Candidate column"]
            if candidate_name == data["Candidate column"]:
                recommendations[idx] = data

                self.redo_stack.append(d)
                self._write_json(recommendations)
                self.get_heatmap()
                return
    
    def _redo_user_action(self):
        if len(self.redo_stack) == 0:
            return
        data = self.redo_stack.pop()
        recommendations = self._load_json()
        for idx, d in enumerate(recommendations):
            candidate_name = d["Candidate column"]
            if candidate_name == data["Candidate column"]:
                recommendations[idx] = data

                self.undo_stack.append(d)
                self._write_json(recommendations)
                self.get_heatmap()
                return
                

    def plot_heatmap(self):
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

        undo_button = pn.widgets.Button(name="Undo", button_type="primary")
        redo_button = pn.widgets.Button(name="Redo", button_type="primary")

        # Subschemas
        if self.ground_truth == "gdc":
            select_rec_groups = pn.widgets.MultiChoice(
                name="Recommendation subschema", options=self.subschemas, width=180
            )
            show_subschema = pn.widgets.Checkbox(name="Show subschema", value=False)
            subschema_col = pn.Column(
                select_rec_groups,
                show_subschema,
            )

        def on_click_accept_match(event):
            self._accept_match()

        def on_click_reject_match(event):
            self._reject_match()
        
        def on_click_undo(event):
            self._undo_user_action()
        
        def on_click_redo(event):
            self._redo_user_action()

        acc_button.on_click(on_click_accept_match)
        rej_button.on_click(on_click_reject_match)
        undo_button.on_click(on_click_undo)
        redo_button.on_click(on_click_redo)

        heatmap_bind = pn.bind(
            self._plot_pane,
            select_column,
            select_rec_groups if self.ground_truth == "gdc" else None,
            n_similar_slider,
            thresh_slider,
            show_subschema if self.ground_truth == "gdc" else False,
            acc_button.param.clicks,
            rej_button.param.clicks,
            undo_button.param.clicks,
            redo_button.param.clicks,
        )

        buttons_down = pn.Row(acc_button, rej_button)
        buttons_redo_undo = pn.Row(undo_button, redo_button)

        column_top = pn.Row(
            select_column,
            subschema_col if self.ground_truth == "gdc" else None,
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

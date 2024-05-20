import json

import altair as alt
import numpy as np
import pandas as pd
import panel as pn
from bdikit.utils import read_gdc_schema
from Levenshtein import distance
from sklearn.cluster import AffinityPropagation

pn.extension("mathjax")
pn.extension("vega")


class SRHeatMapManager:
    def __init__(self) -> None:
        self.json_path = "reduced_scope.json"

        self.rec_table_df = None
        self.rec_list_df = None
        self.rec_cols = None
        self.subschemas = None
        self.rec_cols_gdc = None
        self.clusters = None

    def _load_json(self):
        with open(self.json_path) as f:
            data = json.load(f)
        return data

    def _write_json(self, data):
        with open(self.json_path, "w") as f:
            json.dump(data, f)

    def get_heatmap(self, recommendations):
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

        print(f"Number of clusters: {np.unique(affprop.labels_).shape[0]}\n")
        cluster_names = []
        clusters = {}
        for cluster_id in np.unique(affprop.labels_):
            exemplar = words[affprop.cluster_centers_indices_[cluster_id]]
            cluster = np.unique(words[np.nonzero(affprop.labels_ == cluster_id)])
            cluster_str = ", ".join(cluster)
            print(" - *%s:* %s" % (exemplar, cluster_str))
            cluster_names.append(exemplar)
            clusters[exemplar] = cluster
        self.clusters = clusters

    def _plot_heatmap(self, clusters=[], subschemas=[], threshold=0.5):
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

        base = (
            alt.Chart(heatmap_rec_list)
            .mark_rect()
            .encode(
                y=alt.X("Column:O", sort=None),
                x=alt.X(f"Recommendation:O", sort=None),
                color="Value:Q",
                tooltip=[
                    alt.Tooltip("Column", title="Column"),
                    alt.Tooltip("Recommendation", title="Recommendation"),
                    alt.Tooltip("Value", title="Value"),
                ],
            )
        )
        return pn.pane.Vega(base)

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

        heatmap_bind = pn.bind(
            self._plot_heatmap, select_cluster, select_rec_groups, thresh_slider
        )

        column_left = pn.Column(
            "# Column",
            select_cluster,
            select_rec_groups,
            thresh_slider,
            styles=dict(background="WhiteSmoke"),
        )

        return pn.Row(column_left, pn.Column(heatmap_bind), scroll=True)

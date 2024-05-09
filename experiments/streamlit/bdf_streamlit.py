import os
import sys
import pathlib
script_directory = pathlib.Path(__file__).parent.parent.parent.resolve()
sys.path.append(script_directory)

import pandas as pd

import streamlit as st
from streamlit_agraph import Config, agraph
from bdf_graph import draw_recommandation_graph
from gdc_schema import GDCSchema

st.title("BDF Matcher Demo")

# Database
st.header("1. Database", anchor=False)
st.subheader("Upload the Raw Data", anchor=False)
uploaded_file = st.file_uploader("Upload Your CSV File", type=["csv"])
raw_dataset = None
if uploaded_file:
    st.write("filename:", uploaded_file.name)
    if uploaded_file.type == "text/csv":
        raw_dataset = pd.read_csv(uploaded_file)
        st.dataframe(raw_dataset)

st.subheader("Explore GDC Schema", anchor=False)
schema = GDCSchema()
if "schema_dfs" not in st.session_state:
    schema_dfs = schema.parse_schema_to_df()
    for subschema, df in schema_dfs.items():
        df.insert(0, "Select", False)
        schema_dfs[subschema] = df
    st.session_state.schema_dfs = schema_dfs
subschema = st.selectbox(
    "Select a subschema: ", list(st.session_state.schema_dfs.keys())
)

if "selected_labels" not in st.session_state:
    st.session_state.selected_labels = []


def update_selected_labels():
    st.session_state.selected_labels = edited_df[edited_df.Select][
        "column_name"
    ].to_list()


edited_df = st.data_editor(
    st.session_state.schema_dfs[subschema],
    hide_index=True,
    column_config={
        "Select": st.column_config.CheckboxColumn(required=True),
        "column_values": st.column_config.ListColumn(),
    },
    on_change=update_selected_labels,
)

# Recommendation
st.header("2. Recommendation", anchor=False)
st.subheader("Contrastive Learning", anchor=False)

import torch
from bdi.mapping_algorithms.scope_reducing._algorithms.contrastive_learning.cl_api import \
    ContrastiveLearningAPI


if "cl_api" not in st.session_state:
    st.session_state.cl_api = ContrastiveLearningAPI()


if st.button("Get Recommendations"):
    recommendations, top_k_results = st.session_state.cl_api.get_recommendations(raw_dataset)
    if "recommendation_graph_comp" not in st.session_state:
        st.session_state.recommendation_graph_comp = draw_recommandation_graph(top_k_results)
    if "recommendations" not in st.session_state:
        st.session_state.recommendations = recommendations

# Recommendations
if "recommendations" in st.session_state:
    st.dataframe(schema.get_df_from_recommandations(st.session_state.recommendations))
# Graph
config = Config(
        width=950,
        height=600,
        directed=True,
        physics=True,
        hierarchical=False,
        # **kwargs
    )
if "recommendation_graph_comp" in st.session_state:
    agraph(nodes=st.session_state.recommendation_graph_comp[0],
           edges=st.session_state.recommendation_graph_comp[1],
           config=config)
    # st.write(top_k_results)




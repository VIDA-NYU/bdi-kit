import streamlit as st
import pandas as pd
from gdc_api_v2 import GDCSchema
from gdc_scoring_interface import GPTHelper

st.title('GDC Matcher Demo')

st.header("1. Upload the Raw Data", anchor=False)
uploaded_file = st.file_uploader("Upload Your CSV File", type=["csv"])

raw_dataset = None
if uploaded_file:
    st.write("filename:", uploaded_file.name)
    if uploaded_file.type == "text/csv":
        raw_dataset = pd.read_csv(uploaded_file)
        st.dataframe(raw_dataset)

st.header("2. Explore GDC Schema", anchor=False)
schema = GDCSchema()

if "schema_dfs" not in st.session_state:
    schema_dfs = schema.parse_schema_to_df()
    for subschema, df in schema_dfs.items():
        df.insert(0, "Select", False)
        schema_dfs[subschema] = df
    st.session_state.schema_dfs = schema_dfs



subschema = st.selectbox(
    'Select a subschema: ',
    list(st.session_state.schema_dfs.keys()))

if "selected_labels" not in st.session_state:
    st.session_state.selected_labels = []


def update_selected_labels():
    st.session_state.selected_labels = edited_df[edited_df.Select]["column_name"].to_list()


edited_df = st.data_editor(
    st.session_state.schema_dfs[subschema],
    hide_index=True,
    column_config={"Select": st.column_config.CheckboxColumn(required=True),
                   "column_values": st.column_config.ListColumn()},
    on_change=update_selected_labels)



# Select Labels for CTA

if "gdc_labels" not in st.session_state:
    st.session_state.gdc_labels = set()

# st.dataframe(st.session_state.selected_labels, use_container_width=True)

st.markdown("""Select from dataframe or input labels at the bottom input box as
```
label 1
label 2
label 3...
```
""")

col1, col2 = st.columns(2)
list_input = col1.text_area('Import labels as text (seperate by line):')
labels_csv = col2.file_uploader("Import labels as csv:", type=["csv"])
if labels_csv:
    if labels_csv.type == "text/csv":
        raw_dataset = pd.read_csv(labels_csv)["value"].to_list()
        st.session_state.gdc_labels.update(raw_dataset)


button_col1, button_col2, button_col3 = st.columns(3)
if button_col1.button('Add to GDC labels', use_container_width=True):
    if len(edited_df[edited_df.Select]["column_name"].to_list()) != 0:
        st.session_state.gdc_labels.update(edited_df[edited_df.Select]["column_name"].to_list())
    if list_input != "":
        st.session_state.gdc_labels.update([text.strip() for text in list_input.split("\n")])
if button_col2.button('Clear selection', use_container_width=True):
    st.session_state.gdc_labels.clear()


st.write("Selected labels:")
st.data_editor(st.session_state.gdc_labels)

st.header("3. Match Columns Using CTA", anchor=False)
gpt = GPTHelper(api_key="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")

black_list = ["Case_ID"]
output_dict = {"column_name": [], "gdc_attribute_name": [], "gdc_type": [], "gdc_description": [], "gdc_values": []}
if st.button('Ask CTA'):
    if raw_dataset is None:
        st.warning("Please upload a CSV file to proceed.")
    else:
        progress_text = "CTA start matching columns..."
        my_bar = st.progress(0, text=progress_text)
        col_num = raw_dataset.shape[1]
        for idx, col_name in enumerate(raw_dataset.columns):
            values = raw_dataset[col_name].drop_duplicates().dropna()
            if len(values) > 15:
                rows = values.sample(15).tolist()
            else:
                rows = values.tolist()
            serialized_input = f"{col_name}: {', '.join([str(row) for row in rows])}"
            context = serialized_input

            result = gpt.ask_cta(labels=list(st.session_state.gdc_labels), context=context)
            output_dict["column_name"].append(col_name)
            output_dict["gdc_attribute_name"].append(result)
            if result is not None and result.lower().strip() != "none" and result not in black_list:
                progress_text = f"Column name: {col_name}, Generated column type: {result}"
                properties = schema.get_properties_by_column_name(result)
                output_dict["gdc_type"].append(properties[1])
                output_dict["gdc_description"].append(properties[2])
                output_dict["gdc_values"].append(properties[3])
            else:
                output_dict["gdc_type"].append(None)
                output_dict["gdc_description"].append(None)
                output_dict["gdc_values"].append([])

            my_bar.progress((idx+1)/col_num, text=progress_text)
            
        st.data_editor(pd.DataFrame(output_dict),
                       column_config={"gdc_values": st.column_config.ListColumn()})


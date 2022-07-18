import streamlit as st
from datasets import load_dataset

data = load_dataset("clarin-pl/polemo2-official")
train_subset = data["train"].to_pandas()
validation_subset = data["validation"].to_pandas()
test_subset = data["test"].to_pandas()

header = st.container()
description = st.container()
train_data_section = st.container()
validation_data_section = st.container()
test_data_section = st.container()

with header:
    st.header("abusive-clauses dashboard")

with description:
    desc = (
        "The dataset contains 10000 rows, which are clauses from Polish legal documents. There are two columns: "
        "text and target "
    )
    st.write(desc)

with train_data_section:
    st.table(train_subset.describe())

from pathlib import Path
from typing import Dict # using python 3.8.5

import pandas as pd
import streamlit as st


SPLIT_NAMES = ['train', 'dev', 'test']

# --- Functions ---

flatten_list = lambda main_list: [item for sublist in main_list for item in sublist]


@st.cache
def load_data() -> Dict[str, pd.DataFrame]:
    data = {name:pd.read_csv(f'data/{name}.csv') for name in SPLIT_NAMES}
    return data
DATA_DICT = load_data()

# --- / Functions ----
# --- PAGE  CONTENT ---


st.title("PAC - Polish Abusive Clauses Dataset")

st.header("Dataset Statistics")

st.write(f"Total Samples: {sum([len(df) for _, df in DATA_DICT.items()])}")

st.subheader("Class distribution per data split")
st.bar_chart(pd.DataFrame([df.label.value_counts().rename(k) for k, df in DATA_DICT.items()]))

st.subheader("Number of words per observation")
st.area_chart(
    pd.DataFrame([(
                    df.text.apply(lambda x: len(x.split(' ')))
                    .value_counts()
                    .rename(k)
                    .sort_index()
                    .reindex(list(range(150)))
                    .fillna(0)
                    ) for k, df in DATA_DICT.items()]
                ).T
)

st.subheader("Top 20 common words per data split")
for i, col in enumerate(st.columns(3)):
    with col:
        st.caption(f"Split Name: {SPLIT_NAMES[i].upper()}")
        flat_word_list = flatten_list(DATA_DICT[SPLIT_NAMES[i]].text.apply(lambda x: x.lower().split(' ')).to_list())
        top10_words = 100 * pd.Series(flat_word_list, name='Occurance %').value_counts(normalize=True)[:20]
        st.dataframe(top10_words)
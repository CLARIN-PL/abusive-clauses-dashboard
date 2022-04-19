import re
from pathlib import Path
from typing import Dict # using python 3.8.5

import pandas as pd
import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from unidecode import unidecode


SPLIT_NAMES = ['train', 'dev', 'test']

# --- Functions ---

flatten_list = lambda main_list: [item for sublist in main_list for item in sublist]

count_num_of_characters = lambda text: len(re.sub(r'[^a-zA-Z]', '', unidecode(text)))

count_num_of_words = lambda text: len(re.sub(r'[^a-zA-Z ]', '', unidecode(text)).split(' '))

@st.cache
def load_data() -> Dict[str, pd.DataFrame]:
    data = {name:pd.read_csv(f'/data/{name}.csv') for name in SPLIT_NAMES}
    return data

# --- / Functions ----
# --- PAGE  CONTENT ---
DATA_DICT = load_data()


st.title("PAC - Polish Abusive Clauses Dataset")

st.header("Dataset Description")
st.write(
"""
On the EU and at the national such as the Polish levels, agencies cannot check possible agreements by hand. 
Hence, we took the first step to evaluate the possibility of accelerating this process. 
We created a dataset and machine learning models for partially automating the detection of potentially abusive clauses. 
Consumer protection organizations and agencies can use these resources to make their work more eﬀective and eﬃcient. 
Moreover, consumers can automatically analyze contracts and understand what they agree upon.
"""
)
st.write("Paper: [arxiv](https://arxiv.org/)")
st.write("Github: [github](https://github.com/CLARIN-PL/abusive-clauses-dashboard)")

st.header("Dataset Statistics")

st.write(f'Train Samples: {len(DATA_DICT["train"])}')
st.write(f'Val Samples: {len(DATA_DICT["dev"])}')
st.write(f'Test Samples: {len(DATA_DICT["test"])}') 
st.write(f"Total: {sum([len(df) for _, df in DATA_DICT.items()])}")

st.subheader("Class distribution per data split")
df_class_dist = pd.DataFrame([df['class'].value_counts().rename(k) for k, df in DATA_DICT.items()]).reset_index().rename({'index': 'split_name'}, axis=1)
barchart_class_dist = go.Figure(data=[
    go.Bar(name='BEZPIECZNE_POSTANOWIENIE_UMOWNE', x=SPLIT_NAMES, y=df_class_dist['BEZPIECZNE_POSTANOWIENIE_UMOWNE'].values),
    go.Bar(name='KLAUZULA_ABUZYWNA', x=SPLIT_NAMES, y=df_class_dist['KLAUZULA_ABUZYWNA'].values),
])
barchart_class_dist.update_layout(
    barmode='group', 
    title_text='Barchart - class distribution', 
    xaxis_title='Split name',
    yaxis_title='Number of data points'
)
st.plotly_chart(barchart_class_dist, use_container_width=True)

st.subheader("Number of words per observation")
hist_data_num_words = [df['text'].apply(count_num_of_words).values for df in DATA_DICT.values()]
fig_num_words = ff.create_distplot(hist_data_num_words, SPLIT_NAMES, show_rug=False, bin_size=1)
fig_num_words.update_traces(nbinsx=100, autobinx=True, selector={'type':'histogram'}) 
fig_num_words.update_layout(title_text='Histogram - number of words per observation', xaxis_title='Number of words')
st.plotly_chart(fig_num_words, use_container_width=True)

st.subheader("Character count per observation")
hist_data_num_chars = [df['text'].apply(count_num_of_characters).values for df in DATA_DICT.values()]
fig_num_chars = ff.create_distplot(hist_data_num_chars, SPLIT_NAMES, show_rug=False, bin_size=1)
fig_num_chars.update_traces(nbinsx=100, autobinx=True, selector={'type':'histogram'}) 
fig_num_chars.update_layout(title_text='Histogram - number of characters per observation', xaxis_title='Number of characters')
st.plotly_chart(fig_num_chars, use_container_width=True)

st.subheader("Top 20 common words per data split")
for i, col in enumerate(st.columns(3)):
    with col:
        st.caption(f"Split Name: {SPLIT_NAMES[i].upper()}")
        flat_word_list = flatten_list(DATA_DICT[SPLIT_NAMES[i]].text.apply(lambda x: x.lower().split(' ')).to_list())
        top10_words = 100 * pd.Series(flat_word_list, name='Occurance %').value_counts(normalize=True)[:20]
        st.dataframe(top10_words)

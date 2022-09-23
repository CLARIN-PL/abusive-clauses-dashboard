import streamlit as st

from clarin_datasets.polemo_dataset import PolemoDataset
from clarin_datasets.abusive_clauses_dataset import AbusiveClausesDataset
from clarin_datasets.aspectemo_dataset import AspectEmoDataset
from clarin_datasets.kpwr_ner_datasets import KpwrNerDataset
from clarin_datasets.punctuation_restoration_dataset import (
    PunctuationRestorationDataset,
)
from clarin_datasets.nkjp_pos_dataset import NkjpPosDataset
from clarin_datasets.cst_wikinews_dataset import CSTWikinewsDataset


selected_dataset = st.sidebar.selectbox(
    "Choose a dataset to load",
    (
        "clarin-pl/polemo2-official",
        "laugustyniak/abusive-clauses-pl",
        "clarin-pl/aspectemo",
        "clarin-pl/kpwr-ner",
        "clarin-pl/2021-punctuation-restoration",
        "clarin-pl/nkjp-pos",
        "clarin-pl/cst-wikinews",
    ),
)

if selected_dataset == "clarin-pl/polemo2-official":
    dataset = PolemoDataset()
elif selected_dataset == "laugustyniak/abusive-clauses-pl":
    dataset = AbusiveClausesDataset()
elif selected_dataset == "clarin-pl/aspectemo":
    dataset = AspectEmoDataset()
elif selected_dataset == "clarin-pl/kpwr-ner":
    dataset = KpwrNerDataset()
elif selected_dataset == "clarin-pl/2021-punctuation-restoration":
    dataset = PunctuationRestorationDataset()
elif selected_dataset == "clarin-pl/nkjp-pos":
    dataset = NkjpPosDataset()
elif selected_dataset == "clarin-pl/cst-wikinews":
    dataset = CSTWikinewsDataset()


dataset.load_data()
dataset.show_dataset()
st.caption("https://lepiszcze.ml/ - Polish NLP Leaderboard")

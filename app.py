import re

import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go
import pyperclip
import streamlit as st
from unidecode import unidecode

DATA_SPLITS = ["train", "dev", "test"]


def load_data() -> dict[str, pd.DataFrame]:
    return {data: pd.read_csv(f"data/{data}.csv") for data in DATA_SPLITS}


def flatten_list(main_list: list[list]) -> list:
    return [item for sublist in main_list for item in sublist]


def count_num_of_characters(text: str) -> int:
    return len(re.sub(r"[^a-zA-Z]", "", unidecode(text)))


def count_num_of_words(text: str) -> int:
    return len(re.sub(r"[^a-zA-Z ]", "", unidecode(text)).split(" "))


DATA_DICT = load_data()

header = st.container()
description = st.container()
dataset_statistics = st.container()

with header:
    st.title("PAC - Polish Abusive Clauses Dataset")

with description:
    st.header("Dataset description")
    desc = """
    ''I have read and agree to the terms and conditions'' is one of the biggest lies on the Internet. 
    Consumers rarely read the contracts they are required to accept. We conclude agreements over the Internet daily. 
    But do we know the content of these agreements? Do we check potential unfair statements? On the Internet, 
    we probably skip most of the Terms and Conditions. However, we must remember that we have concluded many more 
    contracts. Imagine that we want to buy a house, a car, send our kids to the nursery, open a bank account, 
    or many more. In all these situations, you will need to conclude the contract, but there is a high probability 
    that you will not read the entire agreement with proper understanding. European consumer law aims to prevent 
    businesses from using so-called ''unfair contractual terms'' in their unilaterally drafted contracts, 
    requiring consumers to accept. 

    Our dataset treats ''unfair contractual term'' as the equivalent of an abusive clause. It could be defined as a 
    clause that is unilaterally imposed by one of the contract's parties, unequally affecting the other, or creating a 
    situation of imbalance between the duties and rights of the parties. 
    
    On the EU and at the national such as the Polish levels, agencies cannot check possible agreements by hand. Hence, 
    we took the first step to evaluate the possibility of accelerating this process. We created a dataset and machine 
    learning models to automate potentially abusive clauses detection partially. Consumer protection organizations and 
    agencies can use these resources to make their work more eﬀective and eﬃcient. Moreover, consumers can automatically 
    analyze contracts and understand what they agree upon. 
    """
    st.write(desc)

with dataset_statistics:
    st.header("Dataset statistics")
    st.subheader("Number of samples in each data split")
    metrics_df = pd.DataFrame.from_dict(
        {
            "Train": DATA_DICT["train"].shape[0],
            "Dev": DATA_DICT["dev"].shape[0],
            "Test": DATA_DICT["test"].shape[0],
            "Total": sum(
                [
                    DATA_DICT["train"].shape[0],
                    DATA_DICT["dev"].shape[0],
                    DATA_DICT["test"].shape[0],
                ]
            ),
        },
        orient="index",
    ).reset_index()
    metrics_df.columns = ["Subset", "Number of samples"]
    st.dataframe(metrics_df)

    latex_df = metrics_df.style.to_latex()
    st.button(label="Copy table to LaTeX", on_click=lambda: pyperclip.copy(latex_df))

    # Class distribution in each subset
    st.header("Class distribution in each subset")
    hist = (
        pd.DataFrame(
            [
                df["label"].value_counts(normalize=True).rename(k)
                for k, df in DATA_DICT.items()
            ]
        )
        .reset_index()
        .rename({"index": "split_name"}, axis=1)
    )
    barchart_class_dist = go.Figure(
        data=[
            go.Bar(
                name="BEZPIECZNE_POSTANOWIENIE_UMOWNE",
                x=DATA_SPLITS,
                y=hist["BEZPIECZNE_POSTANOWIENIE_UMOWNE"].values,
            ),
            go.Bar(
                name="KLAUZULA_ABUZYWNA",
                x=DATA_SPLITS,
                y=hist["KLAUZULA_ABUZYWNA"].values,
            ),
        ]
    )
    barchart_class_dist.update_layout(
        barmode="group",
        title_text="Barchart - class distribution",
        xaxis_title="Split name",
        yaxis_title="Number of data points",
    )
    st.plotly_chart(barchart_class_dist, use_container_width=True)

    # Number of words per observation
    hist_data_num_words = [
        df["text"].apply(count_num_of_words) for df in DATA_DICT.values()
    ]
    fig_num_words = ff.create_distplot(
        hist_data_num_words, DATA_SPLITS, show_rug=False, bin_size=1
    )
    fig_num_words.update_traces(
        nbinsx=100, autobinx=True, selector={"type": "histogram"}
    )
    fig_num_words.update_layout(
        title_text="Histogram - number of characters per observation",
        xaxis_title="Number of characters",
    )
    st.plotly_chart(fig_num_words, use_container_width=True)

    # Number of characters per observation
    hist_data_num_characters = [
        df["text"].apply(count_num_of_characters) for df in DATA_DICT.values()
    ]
    fig_num_chars = ff.create_distplot(
        hist_data_num_characters, DATA_SPLITS, show_rug=False, bin_size=1
    )
    fig_num_chars.update_layout(
        title_text="Histogram - number of characters per observation",
        xaxis_title="Number of characters",
    )
    st.plotly_chart(fig_num_chars, use_container_width=True)

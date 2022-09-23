import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datasets import load_dataset
from sklearn.manifold import TSNE
import streamlit as st

from clarin_datasets.dataset_to_show import DatasetToShow
from clarin_datasets.utils import embed_sentence, PLOT_COLOR_PALETTE


class CSTWikinewsDataset(DatasetToShow):
    def __init__(self):
        DatasetToShow.__init__(self)
        self.dataset_name = "clarin-pl/cst-wikinews"
        self.description = f"""
        Dataset link: https://huggingface.co/datasets/{self.dataset_name}
        """

    def load_data(self):
        raw_dataset = load_dataset(self.dataset_name)
        self.data_dict = {
            subset: raw_dataset[subset].to_pandas() for subset in self.subsets
        }

    def show_dataset(self):
        header = st.container()
        dataframe_head = st.container()
        class_distribution = st.container()
        tsne_projection = st.container()
        with header:
            st.title(self.dataset_name)

        with dataframe_head:
            st.header("First 10 observations of the chosen subset")
            subset_to_show = st.selectbox(
                label="Select subset to see", options=self.subsets
            )
            df_to_show = self.data_dict[subset_to_show].head(10)
            st.dataframe(df_to_show)
            st.text_area(label="LaTeX code", value=df_to_show.style.to_latex())

        class_distribution_df = pd.merge(
            pd.DataFrame(
                self.data_dict["train"]["label"]
                .value_counts(normalize=True)
                .reset_index(drop=False)
                .rename({"index": "class"}, axis="columns")
            ),
            pd.DataFrame(
                self.data_dict["test"]["label"]
                .value_counts(normalize=True)
                .reset_index(drop=False)
                .rename({"index": "class"}, axis="columns")
            ),
            on="class",
        ).rename({"label_x": "train", "label_y": "test"}, axis="columns")

        with class_distribution:
            st.dataframe(class_distribution_df)

        with tsne_projection:
            st.header("t-SNE projection of the dataset")
            subset_to_project = st.selectbox(
                label="Select subset to project", options=self.subsets
            )
            first_sentences = self.data_dict[subset_to_project]["sentence_1"].values
            second_sentences = self.data_dict[subset_to_project]["sentence_2"].values
            labels = self.data_dict[subset_to_project]["label"].values
            first_sentences_embedded = np.array([embed_sentence(x) for x in first_sentences])
            second_sentences_embedded = np.array([embed_sentence(x) for x in second_sentences])
            mean_embeddings = (first_sentences_embedded + second_sentences_embedded) / 2
            reducer = TSNE(
                n_components=2
            )
            transformed_embeddings = reducer.fit_transform(mean_embeddings)
            fig, ax = plt.subplots()
            ax.scatter(
                x=transformed_embeddings[:, 0],
                y=transformed_embeddings[:, 1],
                c=[
                    PLOT_COLOR_PALETTE[i] for i in labels
                ]
            )
            st.pyplot(fig)

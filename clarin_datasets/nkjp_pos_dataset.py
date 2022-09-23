import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datasets import load_dataset
from sklearn.manifold import TSNE
import streamlit as st

from clarin_datasets.dataset_to_show import DatasetToShow
from clarin_datasets.utils import (
    PLOT_COLOR_PALETTE,
    embed_sentence
)


class NkjpPosDataset(DatasetToShow):
    def __init__(self):
        DatasetToShow.__init__(self)
        self.data_dict_named = None
        self.dataset_name = "clarin-pl/nkjp-pos"
        self.description = [
            f"""
            Dataset link: https://huggingface.co/datasets/{self.dataset_name}
            
            NKJP-POS is a part the National Corpus of Polish (Narodowy Korpus Języka Polskiego). 
            Its objective is part-of-speech tagging, e.g. nouns, verbs, adjectives, adverbs, etc. During the creation of 
            corpus, texts of were annotated by humans from various sources, covering many domains and genres. 
            """,
            "Tasks (input, output and metrics)",
            """
            Part-of-speech tagging (POS tagging) - tagging words in text with their corresponding part of speech.

            Input ('tokens' column): sequence of tokens
            
            Output ('pos_tags' column): sequence of predicted tokens’ classes (35 possible classes, described in detail in the annotation guidelines)
            
            Measurements: F1-score (seqeval)
            
            Example:
            
            Input: ['Zarejestruj', 'się', 'jako', 'bezrobotny', '.']
            
            Input (translated by DeepL): Register as unemployed.
            
            Output: ['impt', 'qub', 'conj', 'subst', 'interp']
            """,
        ]

    def load_data(self):
        raw_dataset = load_dataset(self.dataset_name)
        self.data_dict = {
            subset: raw_dataset[subset].to_pandas() for subset in self.subsets
        }
        self.data_dict_named = {}
        for subset in self.subsets:
            references = raw_dataset[subset]["pos_tags"]
            references_named = [
                [
                    raw_dataset[subset].features["pos_tags"].feature.names[label]
                    for label in labels
                ]
                for labels in references
            ]
            self.data_dict_named[subset] = pd.DataFrame(
                {
                    "tokens": self.data_dict[subset]["tokens"],
                    "tags": references_named,
                }
            )

    def show_dataset(self):
        header = st.container()
        description = st.container()
        dataframe_head = st.container()
        class_distribution = st.container()
        tsne_projection = st.container()

        with header:
            st.title(self.dataset_name)

        with description:
            st.header("Dataset description")
            st.write(self.description[0])
            st.subheader(self.description[1])
            st.write(self.description[2])

        with dataframe_head:
            st.header("First 10 observations of the chosen subset")
            subset_to_show = st.selectbox(
                label="Select subset to see", options=self.subsets
            )
            df_to_show = (
                self.data_dict[subset_to_show].head(10).drop("id", axis="columns")
            )
            st.dataframe(df_to_show)
            st.text_area(label="LaTeX code", value=df_to_show.style.to_latex())

        class_distribution_dict = {}
        for subset in self.subsets:
            all_labels_from_subset = self.data_dict_named[subset]["tags"].tolist()
            all_labels_from_subset = [
                x for subarray in all_labels_from_subset for x in subarray
            ]
            all_labels_from_subset = pd.Series(all_labels_from_subset)
            class_distribution_dict[subset] = (
                all_labels_from_subset.value_counts(normalize=True)
                .sort_index()
                .reset_index()
                .rename({"index": "class", 0: subset}, axis="columns")
            )

        class_distribution_df = pd.merge(
            class_distribution_dict["train"],
            class_distribution_dict["test"],
            on="class",
        )

        with class_distribution:
            st.header("Class distribution in each subset")
            st.dataframe(class_distribution_df)
            st.text_area(
                label="LaTeX code", value=class_distribution_df.style.to_latex()
            )
        SHOW_TSNE_PROJECTION = False
        if SHOW_TSNE_PROJECTION:
            with tsne_projection:
                st.header("t-SNE projection of the dataset")
                subset_to_project = st.selectbox(
                    label="Select subset to project", options=self.subsets
                )
                tokens_unzipped = self.data_dict_named[subset_to_project]["tokens"].tolist()
                tokens_unzipped = np.array([x for subarray in tokens_unzipped for x in subarray])
                labels_unzipped = self.data_dict_named[subset_to_project]["tags"].tolist()
                labels_unzipped = np.array([x for subarray in labels_unzipped for x in subarray])
                df_unzipped = pd.DataFrame(
                    {
                        "tokens": tokens_unzipped,
                        "tags": labels_unzipped,
                    }
                )
                tokens_unzipped = df_unzipped["tokens"].values
                labels_unzipped = df_unzipped["tags"].values
                mapping_dict = {name: number for number, name in enumerate(set(labels_unzipped))}
                labels_as_ints = [mapping_dict[label] for label in labels_unzipped]
                embedded_tokens = np.array(
                    [embed_sentence(x) for x in tokens_unzipped]
                )
                reducer = TSNE(
                    n_components=2
                )
                transformed_embeddings = reducer.fit_transform(embedded_tokens)
                fig, ax = plt.subplots()
                ax.scatter(
                    x=transformed_embeddings[:, 0],
                    y=transformed_embeddings[:, 1],
                    c=[
                         PLOT_COLOR_PALETTE[i]
                         for i in labels_as_ints
                    ],
                )
                st.pyplot(fig)

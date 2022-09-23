import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datasets import load_dataset
from sklearn.manifold import TSNE
import streamlit as st

from clarin_datasets.dataset_to_show import DatasetToShow
from clarin_datasets.utils import embed_sentence, PLOT_COLOR_PALETTE


class AspectEmoDataset(DatasetToShow):
    def __init__(self):
        DatasetToShow.__init__(self)
        self.dataset_name = "clarin-pl/aspectemo"
        self.description = [
            f"""
            Dataset link: https://huggingface.co/datasets/{self.dataset_name}
            
            AspectEmo Corpus is an extended version of a publicly available PolEmo 2.0 
            corpus of Polish customer reviews used in many projects on the use of different methods in sentiment 
            analysis. The AspectEmo corpus consists of four subcorpora, each containing online customer reviews from the 
            following domains: school, medicine, hotels, and products. All documents are annotated at the aspect level 
            with six sentiment categories: strong negative (minus_m), weak negative (minus_s), neutral (zero), 
            weak positive (plus_s), strong positive (plus_m).
            """,
            "Tasks (input, output and metrics)",
            """
            Aspect-based sentiment analysis (ABSA) is a text analysis method that 
            categorizes data by aspects and identifies the sentiment assigned to each aspect. It is the sequence tagging 
            task.
            
            "Input ('tokens' column): sequence of tokens"
            
            Output ('labels' column): sequence of predicted tokens’ classes ("O" + 6 possible classes: strong negative (
            a_minus_m), weak negative (a_minus_s), neutral (a_zero), weak positive (a_plus_s), strong positive (
            a_plus_m), ambiguous (a_amb) )
            
            Domain: school, medicine, hotels and products
            
            Measurements:
            
            Example: ['Dużo', 'wymaga', ',', 'ale', 'bardzo', 'uczciwy', 'i', 'przyjazny', 'studentom', '.', 'Warto', 'chodzić', 
            'na', 'konsultacje', '.', 'Docenia', 'postępy', 'i', 'zaangażowanie', '.', 'Polecam', '.'] → ['O', 'a_plus_s', 'O', 
            'O', 'O', 'a_plus_m', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'a_zero', 'O', 'a_plus_m', 'O', 'O', 'O', 'O', 'O', 'O'] 
            """,
        ]

    def load_data(self):
        raw_dataset = load_dataset(self.dataset_name)
        self.data_dict = {
            subset: raw_dataset[subset].to_pandas() for subset in self.subsets
        }

    def show_dataset(self):
        header = st.container()
        description = st.container()
        dataframe_head = st.container()
        class_distribution = st.container()
        most_common_tokens = st.container()
        tsne_projection = st.container()

        with header:
            st.title(self.dataset_name)

        with description:
            st.header("Dataset description")
            st.write(self.description[0])
            st.subheader(self.description[1])
            st.write(self.description[2])

        full_dataframe = pd.concat(self.data_dict.values(), axis="rows")
        tokens_all = full_dataframe["tokens"].tolist()
        tokens_all = [x for subarray in tokens_all for x in subarray]
        labels_all = full_dataframe["labels"].tolist()
        labels_all = [x for subarray in labels_all for x in subarray]

        with dataframe_head:
            st.header("First 10 observations of the chosen subset")
            selected_subset = st.selectbox(
                label="Select subset to see", options=self.subsets
            )
            df_to_show = self.data_dict[selected_subset].head(10)
            st.dataframe(df_to_show)
            st.text_area(label="LaTeX code", value=df_to_show.style.to_latex())

        class_distribution_dict = {}
        for subset in self.subsets:
            all_labels_from_subset = self.data_dict[subset]["labels"].tolist()
            all_labels_from_subset = [
                x for subarray in all_labels_from_subset for x in subarray if x != 0
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
            st.header("Class distribution in each subset (without '0')")
            st.dataframe(class_distribution_df)
            st.text_area(
                label="LaTeX code", value=class_distribution_df.style.to_latex()
            )

        # Most common tokens from selected class (without 0)
        full_df_unzipped = pd.DataFrame(
            {
                "token": tokens_all,
                "label": labels_all,
            }
        )
        full_df_unzipped = full_df_unzipped.loc[full_df_unzipped["label"] != 0]
        possible_options = sorted(full_df_unzipped["label"].unique())
        with most_common_tokens:
            st.header("10 most common tokens from selected class (without '0')")
            selected_class = st.selectbox(
                label="Select class to show", options=possible_options
            )
            df_to_show = (
                full_df_unzipped.loc[full_df_unzipped["label"] == selected_class]
                .groupby(["token"])
                .count()
                .reset_index()
                .rename({"label": "no_of_occurrences"}, axis=1)
                .sort_values(by="no_of_occurrences", ascending=False)
                .reset_index(drop=True)
                .head(10)
            )
            st.dataframe(df_to_show)
            st.text_area(label="LaTeX code", value=df_to_show.style.to_latex())

            with tsne_projection:
                st.header("t-SNE projection of the dataset")
                subset_to_project = st.selectbox(
                    label="Select subset to project", options=self.subsets
                )
                tokens_unzipped = self.data_dict[subset_to_project]["tokens"].tolist()
                tokens_unzipped = np.array([x for subarray in tokens_unzipped for x in subarray])
                labels_unzipped = self.data_dict[subset_to_project]["labels"].tolist()
                labels_unzipped = np.array([x for subarray in labels_unzipped for x in subarray])
                df_unzipped = pd.DataFrame(
                    {
                        "tokens": tokens_unzipped,
                        "labels": labels_unzipped,
                    }
                )
                df_unzipped = df_unzipped.loc[df_unzipped["labels"] != 0]
                tokens_unzipped = df_unzipped["tokens"].values
                labels_unzipped = df_unzipped["labels"].values
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
                        PLOT_COLOR_PALETTE[x]
                        for x in labels_unzipped
                    ],
                )
                st.pyplot(fig)

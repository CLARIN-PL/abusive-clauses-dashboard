import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go
import streamlit as st
from sklearn.manifold import TSNE

from clarin_datasets.dataset_to_show import DatasetToShow
from clarin_datasets.utils import (
    count_num_of_characters,
    count_num_of_words,
)
from clarin_datasets.utils import embed_sentence


class AbusiveClausesDataset(DatasetToShow):
    def __init__(self):
        DatasetToShow.__init__(self)
        self.dataset_name = "laugustyniak/abusive-clauses-pl"
        self.subsets = ["train", "validation", "test"]
        self.description = f"""
        Dataset link: https://huggingface.co/datasets/{self.dataset_name}
        
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

    def load_data(self):
        self.data_dict = {
            subset: pd.read_csv(f"data/{subset}.csv").rename(
                {"label": "target"}, axis="columns"
            )
            for subset in self.subsets
        }

    def show_dataset(self):
        header = st.container()
        description = st.container()
        dataframe_head = st.container()
        word_searching = st.container()
        dataset_statistics = st.container()
        tsne_projection = st.container()

        with header:
            st.title(self.dataset_name)

        with description:
            st.header("Dataset description")
            st.write(self.description)

        with dataframe_head:
            filtering_options = self.data_dict["train"]["target"].unique().tolist()
            filtering_options.append("All classes")

            st.header("First 10 observations of a chosen class")
            class_to_show = st.selectbox(
                label="Select class to show", options=filtering_options
            )
            df_to_show = pd.concat(
                [
                    self.data_dict["train"].copy(),
                    self.data_dict["validation"].copy(),
                    self.data_dict["test"].copy(),
                ]
            )
            if class_to_show == "All classes":
                df_to_show = df_to_show.head(10)
            else:
                df_to_show = df_to_show.loc[df_to_show["target"] == class_to_show].head(
                    10
                )
            st.dataframe(df_to_show)
            st.text_area(label="Latex code", value=df_to_show.style.to_latex())

        with word_searching:
            st.header("Observations containing a chosen word")
            searched_word = st.text_input(
                label="Enter the word you are looking for below"
            )
            df_to_show = pd.concat(
                [
                    self.data_dict["train"].copy(),
                    self.data_dict["validation"].copy(),
                    self.data_dict["test"].copy(),
                ]
            )
            df_to_show = df_to_show.loc[df_to_show["text"].str.contains(searched_word)]
            st.dataframe(df_to_show)
            st.text_area(label="Latex code", value=df_to_show.style.to_latex())

        with dataset_statistics:
            st.header("Dataset statistics")
            st.subheader("Number of samples in each data split")
            metrics_df = pd.DataFrame.from_dict(
                {
                    "Train": self.data_dict["train"].shape[0],
                    "Validation": self.data_dict["validation"].shape[0],
                    "Test": self.data_dict["test"].shape[0],
                    "Total": sum(
                        [
                            self.data_dict["train"].shape[0],
                            self.data_dict["validation"].shape[0],
                            self.data_dict["test"].shape[0],
                        ]
                    ),
                },
                orient="index",
            ).reset_index()
            metrics_df.columns = ["Subset", "Number of samples"]
            st.dataframe(metrics_df)

            latex_df = metrics_df.style.to_latex()
            st.text_area(label="Latex code", value=latex_df)

            # Class distribution in each subset
            st.subheader("Class distribution in each subset")
            target_unique_values = self.data_dict["train"]["target"].unique()
            hist = (
                pd.DataFrame(
                    [
                        df["target"].value_counts(normalize=True).rename(k)
                        for k, df in self.data_dict.items()
                    ]
                )
                .reset_index()
                .rename({"index": "split_name"}, axis=1)
            )
            plot_data = [
                go.Bar(
                    name=str(target_unique_values[i]),
                    x=self.subsets,
                    y=hist[target_unique_values[i]].values,
                )
                for i in range(len(target_unique_values))
            ]
            barchart_class_dist = go.Figure(data=plot_data)
            barchart_class_dist.update_layout(
                barmode="group",
                title_text="Barchart - class distribution",
                xaxis_title="Split name",
                yaxis_title="Number of data points",
            )
            st.plotly_chart(barchart_class_dist, use_container_width=True)
            st.dataframe(hist)
            st.text_area(label="Latex code", value=hist.style.to_latex())

            # Number of words per observation
            st.subheader("Number of words per observation in each subset")
            hist_data_num_words = [
                df["text"].apply(count_num_of_words) for df in self.data_dict.values()
            ]
            fig_num_words = ff.create_distplot(
                hist_data_num_words, self.subsets, show_rug=False, bin_size=1
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
            st.subheader("Number of characters per observation in each subset")
            hist_data_num_characters = [
                df["text"].apply(count_num_of_characters)
                for df in self.data_dict.values()
            ]
            fig_num_chars = ff.create_distplot(
                hist_data_num_characters, self.subsets, show_rug=False, bin_size=1
            )
            fig_num_chars.update_layout(
                title_text="Histogram - number of characters per observation",
                xaxis_title="Number of characters",
            )
            st.plotly_chart(fig_num_chars, use_container_width=True)

            with tsne_projection:
                st.header("t-SNE projection of the dataset")
                subset_to_project = st.selectbox(
                    label="Select subset to project", options=self.subsets
                )
                sentences = self.data_dict[subset_to_project]["text"].values
                reducer = TSNE(
                    n_components=2
                )
                embedded_sentences = np.array(
                    [embed_sentence(text) for text in sentences]
                )
                transformed_embeddings = reducer.fit_transform(embedded_sentences)
                fig, ax = plt.subplots()
                ax.scatter(
                    x=transformed_embeddings[:, 0],
                    y=transformed_embeddings[:, 1],
                    c=[
                        sns.color_palette()[x]
                        for x in self.data_dict[subset_to_project]["target"].map(
                            {
                                "BEZPIECZNE_POSTANOWIENIE_UMOWNE": 0,
                                "KLAUZULA_ABUZYWNA": 1
                            }
                        ).values
                    ],
                )
                st.pyplot(fig)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datasets import load_dataset
from sklearn.manifold import TSNE
import streamlit as st

from clarin_datasets.dataset_to_show import DatasetToShow
from clarin_datasets.utils import embed_sentence, PLOT_COLOR_PALETTE


class KpwrNerDataset(DatasetToShow):
    def __init__(self):
        DatasetToShow.__init__(self)
        self.data_dict_named = None
        self.dataset_name = "clarin-pl/kpwr-ner"
        self.description = [
            f"""
            Dataset link: https://huggingface.co/datasets/{self.dataset_name}
            
            KPWR-NER is a part the Polish Corpus of Wrocław University of Technology (Korpus Języka 
            Polskiego Politechniki Wrocławskiej). Its objective is named entity recognition for fine-grained categories 
            of entities. It is the ‘n82’ version of the KPWr, which means that number of classes is restricted to 82 (
            originally 120). During corpus creation, texts were annotated by humans from various sources, covering many 
            domains and genres. 
            """,
            "Tasks (input, output and metrics)",
            """
            Named entity recognition (NER) - tagging entities in text with their corresponding type.
            
            Input ('tokens' column): sequence of tokens
            
            Output ('ner' column): sequence of predicted tokens’ classes in BIO notation (82 possible classes, described 
            in detail in the annotation guidelines) 
            
            example:
            
            [‘Roboty’, ‘mają’, ‘kilkanaście’, ‘lat’, ‘i’, ‘pochodzą’, ‘z’, ‘USA’, ‘,’, ‘Wysokie’, ‘napięcie’, ‘jest’, 
            ‘dużo’, ‘młodsze’, ‘,’, ‘powstało’, ‘w’, ‘Niemczech’, ‘.’] → [‘B-nam_pro_title’, ‘O’, ‘O’, ‘O’, ‘O’, ‘O’, 
            ‘O’, ‘B-nam_loc_gpe_country’, ‘O’, ‘B-nam_pro_title’, ‘I-nam_pro_title’, ‘O’, ‘O’, ‘O’, ‘O’, ‘O’, ‘O’, 
            ‘B-nam_loc_gpe_country’, ‘O’]
            """,
        ]

    def load_data(self):
        raw_dataset = load_dataset(self.dataset_name)
        self.data_dict = {
            subset: raw_dataset[subset].to_pandas() for subset in self.subsets
        }
        self.data_dict_named = {}
        for subset in self.subsets:
            references = raw_dataset[subset]["ner"]
            references_named = [
                [
                    raw_dataset[subset].features["ner"].feature.names[label]
                    for label in labels
                ]
                for labels in references
            ]
            self.data_dict_named[subset] = pd.DataFrame(
                {
                    "tokens": self.data_dict[subset]["tokens"],
                    "ner": references_named,
                }
            )

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
        labels_all = pd.concat(self.data_dict_named.values(), axis="rows")[
            "ner"
        ].tolist()
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
            all_labels_from_subset = self.data_dict_named[subset]["ner"].tolist()
            all_labels_from_subset = [
                x
                for subarray in all_labels_from_subset
                for x in subarray
                if x != "O" and not x.startswith("I-")
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
            st.header("Class distribution in each subset (without 'O' and 'I-*')")
            st.dataframe(class_distribution_df)
            st.text_area(
                label="LaTeX code", value=class_distribution_df.style.to_latex()
            )

            # Most common tokens from selected class (without 0)
            full_df_unzipped = pd.DataFrame(
                {
                    "token": tokens_all,
                    "ner": labels_all,
                }
            )
            full_df_unzipped = full_df_unzipped.loc[
                (full_df_unzipped["ner"] != "O")
                & ~(full_df_unzipped["ner"].str.startswith("I-"))
            ]
            possible_options = sorted(full_df_unzipped["ner"].unique())
            with most_common_tokens:
                st.header(
                    "10 most common tokens from selected class (without 'O' and 'I-*')"
                )
                selected_class = st.selectbox(
                    label="Select class to show", options=possible_options
                )
                df_to_show = (
                    full_df_unzipped.loc[full_df_unzipped["ner"] == selected_class]
                    .groupby(["token"])
                    .count()
                    .reset_index()
                    .rename({"ner": "no_of_occurrences"}, axis=1)
                    .sort_values(by="no_of_occurrences", ascending=False)
                    .reset_index(drop=True)
                    .head(10)
                )
                st.dataframe(df_to_show)
                st.text_area(label="LaTeX code", value=df_to_show.style.to_latex())
            SHOW_TSNE_PROJECTION = False
            if SHOW_TSNE_PROJECTION:
                with tsne_projection:
                    st.header("t-SNE projection of the dataset")
                    subset_to_project = st.selectbox(
                        label="Select subset to project", options=self.subsets
                    )
                    tokens_unzipped = self.data_dict_named[subset_to_project]["tokens"].tolist()
                    tokens_unzipped = np.array([x for subarray in tokens_unzipped for x in subarray])
                    labels_unzipped = self.data_dict_named[subset_to_project]["ner"].tolist()
                    labels_unzipped = np.array([x for subarray in labels_unzipped for x in subarray])
                    df_unzipped = pd.DataFrame(
                        {
                            "tokens": tokens_unzipped,
                            "ner": labels_unzipped,
                        }
                    )
                    df_unzipped = df_unzipped.loc[
                        (df_unzipped["ner"] != "O")
                        & ~(df_unzipped["ner"].str.startswith("I-"))
                    ]
                    tokens_unzipped = df_unzipped["tokens"].values
                    labels_unzipped = df_unzipped["ner"].values
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
                            PLOT_COLOR_PALETTE[i] for i in labels_as_ints
                        ]
                    )
                    st.pyplot(fig)

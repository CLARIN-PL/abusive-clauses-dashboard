import re
from typing import List

from embeddings.embedding.auto_flair import AutoFlairDocumentEmbedding
from flair.data import Sentence
from numpy import typing as nt
from unidecode import unidecode

embedding = AutoFlairDocumentEmbedding.from_hub("clarin-pl/word2vec-kgr10")

PLOT_COLOR_PALETTE = [
    "#FAEBD7",
    "#00FFFF",
    "#7FFFD4",
    "#000000",
    "#0000FF",
    "#8A2BE2",
    "#A52A2A",
    "#DEB887",
    "#5F9EA0",
    "#7FFF00",
    "#D2691E",
    "#FF7F50",
    "#6495ED",
    "#FFF8DC",
    "#DC143C",
    "#00FFFF",
    "#00008B",
    "#008B8B",
    "#B8860B",
    "#A9A9A9",
    "#006400",
    "#BDB76B",
    "#8B008B",
    "#556B2F",
    "#FF8C00",
    "#9932CC",
    "#8B0000",
    "#E9967A",
    "#8FBC8F",
    "#2F4F4F",
    "#00CED1",
    "#FFD700",
    "#DAA520",
    "#808080",
    "#FF69B4",
    "#4B0082",
    "#CD5C5C",
    "#7CFC00",
    "#F08080",
    "#66CDAA",
]


def flatten_list(main_list: List[List]) -> List:
    return [item for sublist in main_list for item in sublist]


def count_num_of_characters(text: str) -> int:
    return len(re.sub(r"[^a-zA-Z]", "", unidecode(text)))


def count_num_of_words(text: str) -> int:
    return len(re.sub(r"[^a-zA-Z ]", "", unidecode(text)).split(" "))


def embed_sentence(sentence: str) -> nt.NDArray:
    sentence = Sentence(sentence)
    embedding.embed([sentence])
    return sentence.embedding.numpy()

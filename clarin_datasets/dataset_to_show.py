from datasets import load_dataset

from abc import ABC, abstractmethod


class DatasetToShow(ABC):
    @abstractmethod
    def __init__(self):
        self.dataset_name = None
        self.data_dict = None
        self.subsets = ["train", "test"]
        self.description = None

    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def show_dataset(self):
        pass

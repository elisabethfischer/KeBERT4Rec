from abc import ABC, abstractmethod
from multiprocessing import Pool
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, BertConfig, BertModel

"""
Class that converts the content information (e.g. movie categories) to different representations.
It is asserted that the DataFrame from the Dataset is equal to the ML structure.
This means, if there are multiple categories for one product, there is ONE column that contains all 
of these categories with a given separator (default '|')"
"""


class AbstractContentTokenizer(ABC):
    """
    properties:
    @num_classes: number of different content aspects:
        All in one: number of different string in the column
        List: Number of unique categories
        Bert: Number of unique tokens (e.g. separates Children's into children, ' and s)
    @max_seq_len
    """
    def __init__(self, content_column, n_cores=1):
        self.max_len = 1
        self.content_map = self._create_initial_dict(content_column)
        self.n_cores = n_cores

    @abstractmethod
    def _create_initial_dict(self, content: pd.Series) -> Dict[str, int]:
        pass

    @abstractmethod
    def entry_to_id_list(self, content_string: str) -> List[int]:
        pass

    def convert_column_to_ids(self, column: pd.Series) -> pd.Series:
        with Pool(self.n_cores) as p:
            split = np.array_split(column, self.n_cores)
            res = pd.concat(p.map(self.__call__, split))
        return res

    def __call__(self, column: pd.Series) -> pd.Series:
        return column.map(self.entry_to_id_list)

    @property
    def max_seq_len(self):
        """
        Maximal number of items in column, used for static padding
        """
        return self.max_len

    @property
    def num_classes(self):
        """
        number of different content aspects:
            All in one: number of unique strings in the column (as originally)
            List: Number of unique categories
            Bert: Number of unique tokens (e.g. separates Children's into children, ' and s)
        """
        return len(self.content_map)


class FullStringToOneIdContentTokenizer(AbstractContentTokenizer):
    """
    Converts a list of e.g. categories [Action|Drama|Children] to a list of ONE id for this exact category combination,
    e.g. [2]
    """
    def _create_initial_dict(self, content):
        return {s: i for i, s in enumerate(set(content), 1)}

    def entry_to_id_list(self, content_string):
        return [self.content_map[content_string]]


class MultiHotContentTokenizer(AbstractContentTokenizer):
    def __init__(self, content_column, split_sym="|", n_cores=8):
        """
        Converts a list of e.g. categories [Action|Drama|Children] to a list of ids, starting with 1 [3,5,6]
        """
        self.split_sym = split_sym
        super().__init__(content_column, n_cores)

    def split(self, string):
        return string.split(self.split_sym)

    def _create_initial_dict(self, content):
        content = content.unique()
        self.max_len = max(map(len, [self.split(co) for co in content]))
        cats = {x for y in content for x in self.split(y)}
        return {s: i for i, s in enumerate(sorted(cats), 1)}

    def entry_to_id_list(self, content_string):
        ids = [self.content_map[c] for c in self.split(content_string)]
        return ids
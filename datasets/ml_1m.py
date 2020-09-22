from .base import AbstractDataset

import pandas as pd

from datetime import date


class ML1MDataset(AbstractDataset):
    @classmethod
    def code(cls):
        return 'ml-1m'

    @classmethod
    def url(cls):
        return 'http://files.grouplens.org/datasets/movielens/ml-1m.zip'

    @classmethod
    def zip_file_content_is_folder(cls):
        return True

    @classmethod
    def all_raw_file_names(cls):
        return ['README',
                'movies.dat',
                'ratings.dat',
                'users.dat']

    def load_ratings_df(self):
        folder_path = self._get_rawdata_folder_path()
        file_path = folder_path.joinpath('ratings.dat')
        df = pd.read_csv(file_path, sep='::', header=None, engine="python")
        df.columns = ['uid', 'sid', 'rating', 'timestamp']
        c_file_path = folder_path.joinpath('movies.dat')
        mv_df = pd.read_csv(c_file_path, sep='::', header=None)
        mv_df.columns = ['sid', 'c2', 'c1']
        mv_df = mv_df[["sid","c1"]]
        df = pd.merge(df,mv_df)
        return df



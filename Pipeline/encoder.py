import modin.pandas as pd
import numpy as np


class Encoder:
    def __init__(self):
        self._map = {}

    def fit(self, data: pd.Series, encode_nan=False):
        if not encode_nan:
            data = data.dropna()
        data_ = data.unique()
        self._map = {value: i[0] for i, value in np.ndenumerate(data_)}

    @staticmethod
    def is_null(e) -> bool:
        return pd.isna(e)

    def transform(self, data: pd.Series) -> pd.Series:
        encoded = data.apply(lambda x: x if self.is_null(x) else self._map[x])
        return encoded

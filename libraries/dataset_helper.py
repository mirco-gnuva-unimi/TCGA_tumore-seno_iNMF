import os


class DatasetHelper:
    def __init__(self, root: str, filename: str, pandas_library):
        self.root = root
        self.filename = filename
        self.path = os.path.join(root, filename)
        self.name, self.extension = os.path.splitext(filename)
        self.library = pandas_library
        self._dataset = None

    @property
    def dataset(self):
        if self._dataset is None:
            self.load()

        return self._dataset

    @dataset.setter
    def dataset(self, dataset):
        self._dataset = dataset

    def load(self):
        if self._dataset:
            return self._dataset

        dataset = self.library.read_pickle(self.path)

        self.dataset = dataset

        return dataset

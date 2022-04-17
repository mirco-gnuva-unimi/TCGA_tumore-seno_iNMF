from umap import UMAP
import pandas


def umap(dataset: pandas.DataFrame, n_components: int = None, n_neighbors: int = 15, low_memory: bool = False):
    assert(isinstance(dataset, pandas.DataFrame))
    assert(isinstance(n_components, int) or n_components is None)

    umap = UMAP(n_components=n_components, random_state=0, n_neighbors=n_neighbors, low_memory=low_memory)

    projection = umap.fit_transform(dataset)
    projection = pandas.DataFrame(projection, index=dataset.index)

    return projection
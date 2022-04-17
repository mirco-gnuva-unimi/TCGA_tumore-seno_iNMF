from sklearn.decomposition import PCA
from tqdm.notebook import tqdm


# Given a dataset and, optionally, the number of components; computes PCA and returns a list of pairs, where each pair is: <feature name>, <explained_variance_ratio>
def compute_pca(dataset, n_components=None):
    pca = PCA(n_components=n_components, random_state=0)
    pca.fit(dataset)
    pca_results = [(pca.feature_names_in_[i], pca.explained_variance_ratio_[i]) for i in range(len(pca.components_))]

    return pca_results


# Selects features to reach <ratio>
def cumulative_ratio(pca_results, ratio):
    selected_components = []

    i = 0

    while sum([component_ratio[1] for component_ratio in selected_components]) <= ratio:
        selected_components.append(pca_results[i])
        i += 1

    return selected_components


# Select the <n_components> first components from PCA
def select_components(pca_results, n_components):
    selected_components = pca_results[:n_components]

    return selected_components


def pca(dataset, n_components: int, param: float):
    results = compute_pca(dataset, n_components)
    if param <= 1:
        selected_pairs = cumulative_ratio(results, param)
    else:
        selected_pairs = select_components(results, param)

    components = [pair[0] for pair in selected_pairs]
    return dataset[components]


def run(datasets, n_components: int, param: float, threshold: int, show_bar: bool):
    if param == 0:
        return datasets

    if not isinstance(datasets, dict):
        return pca(datasets, n_components, param)

    generator = tqdm(datasets.items()) if show_bar else datasets.items()
    for name, dataset in generator:
        if show_bar:
            generator.set_description(f'Computing PCA on {name}')

        if len(dataset.columns) >= threshold:
            datasets[name] = pca(dataset, n_components, param)

        if show_bar:
            generator.set_description('PCA completed')

    return datasets

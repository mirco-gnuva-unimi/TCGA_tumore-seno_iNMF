from more_itertools import powerset


def same_norm(subset) -> bool:
    A = 'miRNA_vst' in subset
    B = 'miRNA_mor' in subset
    C = 'mRNA_vst' in subset
    D = 'mRNA_mor' in subset

    return (not A and not C) or (not B and not D and (C or A))


def get_subsets(datasets: dict) -> list:
    keys = datasets.keys()
    keys_subsets = [subset for subset in powerset(keys) if len(subset) > 1 and same_norm(subset)]

    return [{key: dataset for key, dataset in datasets.items() if key in keys_subset} for keys_subset in keys_subsets]

from nmf import integrative_nmf
from nmf import nmf
import contextlib


def inmf(datasets: dict, n_threads: int, use_gpu: bool, n_components: int, tol: float, lam: float, algo: str,
         batch_max_iter: int = 200, random_state: int = 0,  log: bool  = False):
    assert isinstance(datasets, dict)
    assert isinstance(n_threads, int)
    assert isinstance(use_gpu, bool)
    assert isinstance(n_components, int)
    assert isinstance(tol, float)
    assert isinstance(lam, float)
    assert isinstance(algo, str)
    assert isinstance(batch_max_iter, int)

    matrices = [dataset.transpose().to_numpy() for dataset in datasets.values()]

    hw_parameter = {'use_gpu': True} if use_gpu else {'n_jobs': n_threads}

    if log:
        H, W, V, err = integrative_nmf(matrices, n_components=n_components, algo=algo, mode='batch', tol=tol,
                                       random_state=random_state, lam=lam, batch_max_iter=batch_max_iter,
                                       **hw_parameter)
    else:
        with contextlib.redirect_stdout(None):
            H, W, V, err = integrative_nmf(matrices, n_components=n_components, algo=algo, mode='batch', tol=tol,
                                           random_state=random_state, lam=lam, batch_max_iter=batch_max_iter,
                                           **hw_parameter)

    return H, W, V, err

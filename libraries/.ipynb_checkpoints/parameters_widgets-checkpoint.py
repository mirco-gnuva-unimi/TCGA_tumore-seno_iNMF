import numpy as np
from ipywidgets import widgets
from IPython.display import display
import torch

int64_info = np.iinfo(np.int64)
max_int = int64_info.max

# Threads number
n_threads = widgets.IntSlider(value=-1, min=-1, max=64, step=1, description='CPU threads:', continuous_update=False)

# Use GPU
use_gpu = widgets.Checkbox(value=False, description='Use GPU')

# Convergence tolerance
tol = widgets.BoundedFloatText(value=1e-4, min=0, max=max_int, description='Convergence tolerance:',
                               continuous_update=False)

# Regularization parameter
lam = widgets.BoundedFloatText(value=5, min=0, max=max_int, description='Regularization:', continuous_update=False)

# Max iterations per batch
batch_max_iter = widgets.IntSlider(value=200, min=0, max=1000, step=1,
                                   description='Max iteration/batch:, continuous_update=False')

# Factorization components
n_components = widgets.BoundedIntText(value=3, description='Factorization components:', disabled=False, min=1,
                                      max=max_int, continuous_update=False)

# Algorithm
algo = widgets.Dropdown(options=['mu', 'halsvar', 'bpp'], value='halsvar', description='Algorithm:')

# Threaded Pandas
threaded = widgets.Checkbox(value=False, description='Threaded DataFrame manipulation')

oversample = widgets.Checkbox(value=False, description='Oversample')


# True: CUDA library available
def gpu_available() -> bool:
    return torch.cuda.is_available()


# Displays iNMF related parameters
def display_inmf_parameters():
    display(tol)
    display(lam)
    display(batch_max_iter)
    display(n_components)
    display(algo)


# Displays hardware related parameters
def display_hw_parameters():
    if gpu_available():
        display(use_gpu)
        display(n_threads)
    else:
        display(n_threads)

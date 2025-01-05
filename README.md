# NUMAP

<p align="center">

[//]: # (This is the official PyTorch implementation of NUMAP, a new and generalizable UMAP implementation, from the paper ["Generalizable Spectral Embedding with Applications to UMAP"]&#40;&#41;.<br>)
This is the official PyTorch implementation of NUMAP, a new and generalizable UMAP implementation.

See our [GitHub repository](https://github.com/shaham-lab/NUMAP) for more information and the latest updates.

[//]: # (## Installation)

[//]: # (You can install the latest package version via)

[//]: # (```bash)
[//]: # (pip install spectralnet)
[//]: # (```)

NUMAP can be used to visualize many types of data in a low-dimensional space, while enabling a simple out-of-sample extension.
One application of NUMAP is to **visualize time-series data**, and help understand the process in a given system.
For example, the following figure shows the transition of a set of points from one state to another, using NUMAP.
In a biological point of view, this can be viewed as a simplified simulation of the cellular differentiation process.

[//]: # (github)
<img src="figures\NUMAP_timesteps_transition_1color.png">

[//]: # (pypi)
[//]: # (<img src="https://github.com/shaham-lab/NUMAP/raw/main/figures/NUMAP_timesteps_transition_1color.png">)

The package is based on UMAP and [**GrEASE (Generalizable and Efficient Approximate Spectral Embedding)**](https://github.com/shaham-lab/GrEASE).
It is easy to use and can be used with any PyTorch dataset, on both CPU and GPU.
The package also includes a test dataset and a test script to run the model on the 2 Circles dataset.

The incorporation of GrEASE enables preservation of both **local and global structures** of the data, as UMAP,
with the new capability of out-of-sample extension.

[//]: # (github)
<img src="figures\intro_fig_idsai_colored.png">
    
[//]: # (pypi)
[//]: # (<img src="https://github.com/shaham-lab/NUMAP/raw/main/figures/intro_fig_idsai_colored.png">)

## Installation
To install the package, simply use the following command:

```bash
pip install numap
```

## Usage

The basic functionality is quite intuitive and easy to use, e.g.,

```python
from numap import NUMAP

numap = NUMAP(n_components=2)  # n_components is the number of dimensions in the low-dimensional representation
numap.fit(X)  # X is the dataset and it should be a torch.Tensor
X_reduced = numap.transfrom(X)  # Get the low-dimensional representation of the dataset
Y_reduced = numap.transform(Y)  # Get the low-dimensional representation of a test dataset

```

You can read the code docs for more information and functionalities.<br>

## Running examples

In order to run the model on the 2 Circles dataset, you can either run the file, or using the command-line command:<br>
`python tests/run_numap.py`<br>
This will run NUMAP and UMAP on the 2 Circles dataset and plot the results.




[//]: # (## Citation)

[//]: # ()
[//]: # (```)

[//]: # ()
[//]: # (@inproceedings{shaham2018,)

[//]: # (author = {Uri Shaham and Kelly Stanton and Henri Li and Boaz Nadler and Ronen Basri and Yuval Kluger},)

[//]: # (title = {SpectralNet: Spectral Clustering Using Deep Neural Networks},)

[//]: # (booktitle = {Proc. ICLR 2018},)

[//]: # (year = {2018})

[//]: # (})

[//]: # ()
[//]: # (```)

# NUMAP

<p align="center">

[//]: # (    <img src="https://github.com/shaham-lab/SpectralNet/blob/main/figures/twomoons.png">)

This is the official PyTorch implementation of NUMAP from the paper ["Generalizable Spectral Embedding with Applications to UMAP]().<br>

[//]: # (## Installation)

[//]: # (You can install the latest package version via)

[//]: # (```bash)
[//]: # (pip install spectralnet)
[//]: # (```)

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

In order to run the model on the moon dataset, you can either run the file, or using the command-line command:<br>
`python -m examples.reduce_moon`<br>
This will run the model on the moon dataset and plot the results.

The same can be done for the circles dataset:<br>
`python -m examples.reduce_circles`<br>




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

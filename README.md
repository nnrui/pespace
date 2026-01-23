# `pespace`: A tool of response generation and likelihood evaluation for space-borne gravitational wave detectors

![last commit](https://img.shields.io/github/last-commit/nnrui/pespace)
[![code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> [!Warning]
> This is an experimental project under active development. The design and APIs are not stable and may change frequently.

This package can be used to generate detector responses and evaluate the likelihood function under the stationary Gaussian noise assumption for space-borne detectors, with a focus on
parameter estimation of massive black hole binaries.
Core computations are implemented with `taichi-lang`, enabling automatic differentiation and hardware acceleration across multiple architectures. 
More details can be found in the [paper]() or the [document](). 

## Installation
Install from PyPI:
```bash
pip install pespace
```
Install the latest or specific commit version:
```bash
# install the latest development version
pip install git+https://github.com/nnrui/pespace
# install a specific commit
pip install git+https://github.com/nnrui/pespace@<commit-hash>
```

## Usage
The basic functionality of generating detector responses is domanstrated in the [tutorial](). Example scripts of the full Bayesian parameter estimation for a massive black hole binary merger signal can be found [here]() for a single LISA-like detector, and [here]() for the LISA-Taiji-Tianqin network.

## Similar packages
If `pespace` cannot meet your needs, you may find other packages for similar functionality (welcome to open issues or pull requests if you have more):
- [LISA-Black-Hole](https://github.com/eXtremeGravityInstitute/LISA-Black-Hole)
- [lisabeta](https://gitlab.in2p3.fr/marsat/lisabeta)
- [bbhx](https://github.com/mikekatz04/BBHx)

## Contact
The author strive to make this project easy-to-use and maintainable. But the author's experience and knowledge in software engineering is limited. Any feedback, comments, and suggestions are greatly appreciated. Feel free to open issues or contact.

## Citation
If you think this package is useful, please consider cite [arxiv: 2601.xxxx]().

The development of this package depends on many previous works including:
- The frequency domain response model: [Sylvain Marsat, John G. Baker, arxiv: 1806.10734](https://arxiv.org/abs/1806.10734);

Please cite the original works for the corresponding modules you have used.
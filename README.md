# Computational Package for Molecular-Beam Holographic Lithography (MBHL)
[![DOI](https://zenodo.org/badge/652728964.svg)](https://doi.org/10.5281/zenodo.14986964)

## Installation
The codes require `Python3`, `shapely`, `ase` to run. For post-processing of 3D models,
`Blender` and `Gwyddion` are required.

We recommend using `miniconda` to initialize a working environment:
```bash
conda create -n mbhl "python>=3.8" pip "numpy<2.0" ase shapely jupyterlab
conda activate mbhl
git clone https://github.com/alchem0x2A/nanolitho.git
pip install -e nanolitho
```

## Reproducing the figures
The Jupyter Notebooks for reproducing the figures in the manuscript can be found under `manuscript_figures`
directory:

```bash
conda activate mbhl
cd nanolitho
jupyter lab &
```
To continue, open the corresponding notebooks in the jupyter browser window and run the experiments.

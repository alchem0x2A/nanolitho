# Computational Package for Molecular-Beam Holographic Lithography (MBHL)

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

## Tutorial
An interactive tutorial for the MBHL package can be found at (https://mybinder.org/v2/gh/alchem0x2A/mbhl-public-demo/main). 
The source codes are in (https://github.com/alchem0x2A/mbhl-public-demo).

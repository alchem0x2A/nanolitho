#!/usr/bin/env python

from distutils.core import setup
from warnings import warn

from setuptools import find_packages
from setuptools.command.install import install


setup(
    name="mbhl",
    version="0.0.1",
    python_requires=">=3.8",
    description="Simulation Package for Molecular Beam Holographic Lithography (MBHL)",
    author="Tian Tian",
    author_email="tian.tian@ualberta.ca",
    url="https://github.com/alchem0x2A/nanolitho",
    packages=find_packages(),
    install_requires=["ase>=3.22.0", "numpy>=1.23", "packaging>=20.0", "shapely>=2.0"],
)

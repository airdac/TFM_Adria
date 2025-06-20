# Distance-based Dimensionality Reduction for Big Data

This directory contains the code written and the data used during the master's thesis titled **Distance-based dimensionality reduction for big data**.

## Project Information

- **Author**: Adrià Casanova Lloveras
- **Program**: Master's Degree in Data Science
- **Institution**: Facultat d'Informàtica de Barcelona (FIB), Universitat Politècnica de Catalunya (UPC) - BarcelonaTech
- **Supervisor**: Pedro F. Delicado Useros (Department of Statistics and Operations Research)
- **Co-supervisor**: Cristian Pachón García (Department of Statistics and Operations Research)
- **Defense Date**: July 3rd, 2025
- **Full Project Repository**: https://github.com/airdac/TFM_Adria

## Directory Structure

The structure of this directory is as follows:

```
├── __pycache__/       # Python cache
├── benchmark/         # Code, data and figures to benchmark divide-and-conquer DR
├── Isomap/            # Code, data and figures to analyze divide-and-conquer Isomap
├── old_figures/       # Figures of early analyses
├── results/           # Figures and text data of more early analyses
├── SMACOF/            # Code, data and figures to analyze divide-and-conquer SMACOF
├── t-SNE/             # Code, data and figures to analyze divide-and-conquer t-SNE
├── __init__.py        # Python package initialization file
├── d_and_c.py         # Code implementing divide-and-conquer DR
├── examples.py        # Methods to test and benchmark divide-and-conquer DR
├── methods.py         # Implemented DR methods
├── MNIST_1000.pkl     # Random subset of 1000 images from MNIST for testing
├── MNIST_5000.pkl     # Random subset of 5000 images from MNIST for testing
├── MNIST.RData        # EMNIST dataset
├── private_d_and_c.py # Private methods for d_and_c.py
├── private_lmds.py    # Private methods for methods.local_mds()
├── README.md          # This file
└── utils.py           # Helper functions
```

## Usage
To execute divide-and-conquer DR, run d_and_c.divide_conquer() with the appropiate arguments. Specifically, method has to be an object of the methods.DRMethod class.
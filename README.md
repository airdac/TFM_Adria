# Distance-based dimensionality reduction for big data
Dimensionality reduction aims to project a data set into a low-dimensional space. Many techniques have been proposed, most of them based on the inter-individual distance matrix. When the number of individuals is really large, the use of distance matrices is prohibitive. There are algorithms that extend MDS (a classical dimensionality reduction method based on distances) to the big data setting. In this TFM, we adapt these algorithms to any generic distance-based dimensionality reduction method.

## Project Information

- **Author**: Adrià Casanova Lloveras
- **Program**: Master's Degree in Data Science
- **Institution**: Facultat d'Informàtica de Barcelona (FIB). Universitat Politècnica de Catalunya (UPC) - BarcelonaTech
- **Supervisor**: Pedro F. Delicado Useros (Department of Statistics and Operations Research)
- **Co-supervisor**: Cristian Pachón García (Department of Statistics and Operations Research)
- **Defense Date**: July 3rd, 2025

### Usage
To execute divide-and-conquer DR, run d_and_c.divide_conquer() with the appropiate arguments. Specifically, method has to be an object of the methods.DRMethod class. Currently implemented DR methods are SMACOF, LMDS, Isomap and t-SNE, although new ones can easily be added to the methods.DRMethod class by the user.
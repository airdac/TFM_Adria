# MDS on the GPU with OOS Extension - Code

This directory contains the code for our GPU implementation of MDS. 

## Contents
- `mds_demo.py` contains the methods of the MDS algorithm, including a setup method which is used to initialize and demo the MDS algorithm.
- `synthetic.py` contains a generator for a synthetic dataset with 4 different classes that is used to demo the MDS algorithm.

## Setup
1. Ideally be on a Linux operating system and have a CUDA capable GPU.
2. Have python and pip installed.
3. Optionally create a virtual environment and activate it.
   - `python -m venv venv`
   - `source venv/bin/activate`
4. Install dependencies.
   - `pip install -r requirements.txt`

## Run
6. Run the mds_demo.py script
   - `python mds_demo.py`
   You can freely adapt the n_samples_ref, n_samples_batch and n_batches parameters to test the algorithm with different reference set sizes and batch sizes.

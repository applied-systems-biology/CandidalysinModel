# CandidalysinModel


## Repository Structure

- `fitting.py`: This file includes a collection of functions related to model fitting. It encompasses an abstract class for fitting models, computing validation intervals, and objective functions.

- `model.py`: The core model definition for the ODE models used.

- `model_utils.py`: This utility file provides wrapper functions for model fitting. Key features include simulating synthetic data for testing and evaluating the likelihood function of models.

- `preprocessing.py`: Preprocessing of data for fitting. This file contains methods and utilities for data cleaning, transformation, and preparation, ensuring that the data is in the right format for fitting.

- `publication_figures.ipynb`: A Jupyter Notebook for generating all the tables and figures given in the publication.

- `requirements.txt`: This file lists all the dependencies required to set up a Conda environment for running the scripts and the Jupyter Notebook included in this repository. 

- `script_screening.py`: This script is dedicated to conducting parameter fitting on the data. It includes functionalities for parameter screening and optimization to enhance model accuracy and performance.

## Installation

1. Clone the repository:
``git clone [repository-url]``

2. Set up a Conda environment:
``conda create --name [env-name] --file requirements.txt``

3. Activate the environment:
``conda activate [env-name]``

## License

This project is licensed under BSD - see the LICENSE file for details.

## Data availability

Data can be accessed under [ASBDATA](https://asbdata.hki-jena.de/ValentineEtAl2023_Candidalysin_mBio)

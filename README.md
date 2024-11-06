# Bayesian Causal Discovery for Policy Decision Making

This repository contains code for performing Bayesian causal discovery to inform policy decision-making, as described in the paper "Bayesian Causal Discovery for Policy Decision Making". The analysis uses Bayesian networks to discover causal pathways and estimates their impact in policy contexts such as educational outcomes in Australia.

## Overview

The main analysis is in the Python Jupyter notebook file `synthetic_data_analysis.ipynb`, which includes integration with R's [BiDAG]([https://cran.r-project.org/web/packages/bidag/index.html](https://cran.r-project.org/web/packages/BiDAG/index.html)) library to perform Partition MCMC for Bayesian structure learning. The repository also contains supporting files providing necessary functions for data processing, Bayesian inference, and result evaluation.

## Installation

### Prerequisites
- Python 3.9 or later
- R (version 3.5 or later)

### Installation Instructions

1. Clone the repository:

   ```sh
   git clone https://github.com/human-technology-intitute/bayesian-causal-policy.git
   cd bayesian-causal-discovery
   ```

2. Create a virtual environment and activate it:

   ```sh
   python -m venv env
   source env/bin/activate  # On Windows, use `env\Scripts\activate`
   ```

3. Install the Python dependencies:

   ```sh
   pip install -r requirements.txt
   ```

4. Install the R dependencies. Run R and execute the following:

   ```R
   install.packages("bidag")
   install.packages("rpy2")
   ```

`rpy2` is used to call R functions from Python.

## Running the Analysis

To run the analysis:

1. Make sure R is installed and available in your system PATH.

2. Open the Jupyter notebook:

   ```sh
   jupyter notebook synthetic_data_analysis.ipynb
   ```

3. Run all cells to generate synthetic data, perform Bayesian causal discovery, and analyze the results.

The notebook demonstrates the use of Python for data generation and pre-processing and calls R functions from `BiDAG` for structure learning.

## File Structure

- `synthetic_data_analysis.ipynb`: Jupyter notebook for running the main analysis.
- `inference/`: Directory containing scripts for supporting Bayesian inference.
- `results/`: Directory for storing output data and analysis results.
- `scores/`: Directory containing scoring metric calculation scripts (BGe score).
- `utils/`: Directory with utility functions for analysis.
- `README.md`: Documentation for setting up and running the project.
- `requirements.txt`: List of Python dependencies.

## Dependencies

### Python
- `numpy` (>=1.21.0)
- `pandas` (>=1.3.0)
- `matplotlib` (>=3.4.2)
- `rpy2` (>=3.4.5)
- `networkx` (>=2.5)
- `seaborn` (>=0.11.0)
- `statsmodels` (>=0.12.0)
- `scikit-learn` (>=0.24.0)
- `pgmpy` (>=0.1.13)
- `pickle` 
- `os` 

### R
- `bidag` (>=2.1.4)

## References
- Bayesian Causal Discovery for Policy Decision Making

## License
This project is licensed under the Apache 2.0 License.


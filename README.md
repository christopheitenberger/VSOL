# Versioning System for Online Learning systems (VSOL)

## Description

This repository contains the code of the master thesis titled
['Versioning of Model State Evolution of Machine Learning Models in Online Learning Settings'](https://doi.org/10.34726/hss.2024.108900),
which was written at Vienna University of Technology.

Eitenberger, C. (2024). Versioning of Model State Evolution of Machine Learning Models in Online Learning Settings [Diploma Thesis, Technische Universität Wien]. reposiTUm. https://doi.org/10.34726/hss.2024.108900

VSOL stores neural network weights in an online learning setting.
Restoring previously used versions may be necessary to understand previous decisions of the neural network.
Understanding these decisions can be required by the GDPR’s ''right to explain''' or other
legal claims.
VSOL is designed for fast execution and minimal impact on the error rate and storage space.
It offers one lossless and six lossy settings, using various compression approaches.

The repository contains the code used to evaluate the VSOL and to visualize the results
as well as the results themselves.
The VSOL was tested under constant virtual data drift, simulated through introducing an unseen label
from a static, not online specific data set.
A convolutional as well as a long short-term memory neural network were used for evaluation.
Fail safety is not implemented since it is not required to measure the effect of the VSOL,
which is the focus of the thesis.
Hence, the implementation focuses on executing the evaluation and is not failsafe.

The machine learning framework used is 'Keras' since it features well-defined interfaces
for the learning process, increasing the easy of integrating the VSOL into it.

## Installation

To use the VSOL, the following steps are required

* install python (used version 3.10.11)
* install [requirements file](requirements.txt)

### Download Training Data for Evaluation of VSOL

To evaluate the VSOL, additional training data is require:

* download the following zip files
    * [AG News Dataset](https://www.kaggle.com/code/ishandutta/ag-news-classification-lstm/input)
    * [MNIST Dataset](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv/data)
* Create the following `data` folder structure and move the files accordingly:

```
+-- data
    +-- ag_news
        +-- test.csv
        +-- train.csv
    +-- mnist
        +-- minst_text.csv
        +-- mnist_tain.csv
+-- notebooks 
...
```

## Usage

To execute the VSOL evaluation, use the jupyter notebook
[evaluation_drift_simulation_and_vsol_configurations](./notebooks/evaluation_drift_simulation_and_vsol_configurations.ipynb)

To integrate the VSOL into an existing project, follow the tutorial code
[tutorial_for_using_vsol](./src/tutorial_for_using_vsol.py).
The configuration options and their compression ratios for VSOL are listed in
[performance_requirement_settings_data.py](./src/compression_pipeline_test_runner/performance_requirement_settings_data.py)


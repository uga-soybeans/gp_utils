# gp_utils: UGA Soybean Genomic Prediction Projects Shared Codebase

## Overview

gp_utils is a codebase containing the core genomic prediction algorithms and supporting functionalities developed at UGA soybean breeding program. It can be used as an independent repo or as a supporting package.

### File organization

```
gp_utils/  
├─ data/    (***CONTAINS TESTING DATA. IGNORED BY GIT***)
├─ gp_utils/     (main source code folder) 
    ├─ evaluations/     (source code for performance metrics)
        ├─ __init__.py
        └─ metrics.py   (defines performance metrics)
    ├─ models/          (source code for GP algorithms and initializations)
        ├─ __init__.py
        └─ models.py    (GP algorithms and initializations)
    ├─ pipeline/        (source code training pipelines)
        ├─ __init__.py
        └─ pipeline.py  (training pipeline initialization and training, including grid search)
    ├─ preprocessing/   (source code for loading raw genotype files)
        ├─ __init__.py
        └─ str2num.py   (loading raw genotypes. Support multiple encodings)
    ├─ reducers/        (source code for feature selection algorithms and initializations)
        ├─ __init__.py
        └─ reducers.py  (Feature selectors and initializations)
    ├─ simCross/        (source code for progeny genotype simulation)
        ├─ __init__.py
        └─ simCross.py  (reading genetic map and simulating progeny genotypes)
    ├─ __init__.py
    └─ utils.py (top-level utils for checking R environment)
├─ notebooks/       (Jupyter notebooks)
    └─ example_train.ipynb  (Demonstrating training and grid search. Requires user-provided data/ folder)
├─ .env (IGNORED BY GIT. Contains R_HOME)
├─ .gitignore
├─ LICENSE
├─ pyproject.toml
├─ r_reqruiements.R
├─ README.md (this file)
└─ requirements.txt
```

## Use as an independent repository.

Clone the repo with 
```
git clone https://github.com/uga-soybeans/gp_app.git
```
### Python dependencies

Please make sure **Python (>=3.11)** is installed on your machine.

Required Python packages:
* feature_engine==1.9.3
* numpy==2.2.5
* pandas==2.2.3
* rpy2==3.4.5
* scikit-learn==1.6.1
* scipy==1.15.2

You can install necessary dependencies with 
```
pip install -r requirements.txt
```
### R dependencies

```gp_utils``` interfaces with the R programming language using the ```rpy2``` package (version 3.4.5).

Please make sure **R (version 4.4.1 preferred)** is installed on your machine, and set the environment variable ```R_HOME``` correctly.

Required R packages:
* rrBLUP
* EMMREML
* bWGR
* qtl

## Use as a supporting package

Alternatively, you can install ```gp_utils``` as a package via

```
pip install git+https://github.com/uga-soybeans/gp_utils.git@main
```

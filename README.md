# gp_utils
UGA Soybean Genomic Prediction Projects Shared Codebase.
You can install ```gp_utils``` as a package via

```pip install git+https://github.com/uga-soybeans/gp_utils.git@main```

### R dependencies

```gp_utils``` interfaces with the R programming language using the ```rpy2``` package (version 3.4.5).

Please make sure **R (>=4.0)** is installed on your machine, and set the environment variable ```R_HOME``` correctly.

Required R packages:
* rrBLUP
* EMMREML
* bWGR
* qtl

### Python dependencies

Please make sure **Python (>=3.11)** is installed on your machine.

Required Python packages:
* numpy==2.2.5
* pandas==2.2.3
* rpy2==3.4.5
* scikit-learn==1.6.1
* scipy==1.15.2

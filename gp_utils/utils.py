import os
import shutil
# import rpy2.robjects.packages as rpackages

REQUIRED_R_PACKAGES = [
    'rrBLUP',
    'EMMREML',
    'bWGR',
    'Matrix',
    'qtl'
]

def check_r_environment():
    # Check if R is installed
    if shutil.which("R") is None:
        raise EnvironmentError(
            "R is not installed or not in your PATH.\n"
            "Please install R (https://cran.r-project.org/) before using gp_utils.\n"
            f"The following R packages are required: {', '.join(REQUIRED_R_PACKAGES)}."
        )
    
    # Check if R_HOME is set
    if os.environ.get("R_HOME") is None:
        raise EnvironmentError(
            "Please set the R_HOME environment variable.\n"
            f"The following R packages are required: {', '.join(REQUIRED_R_PACKAGES)}."
        )

    # # Check R packages
    # missing = []
    # for pkg in REQUIRED_R_PACKAGES:
    #     if not rpackages.isinstalled(pkg):
    #         missing.append(pkg)

    # if missing:
    #     raise ImportError(
    #         f"The following R packages are required but not installed: {', '.join(missing)}. "
    #         f"You can install them in R using:\n"
    #         f"    install.packages(c({', '.join(repr(p) for p in missing)}))"
    #     )

def ensure_r_ready():
    """Call this function before any R-interfacing operations."""
    check_r_environment()

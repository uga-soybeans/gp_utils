__version__ = "0.1.0"

try:
    from .utils import ensure_r_ready
    ensure_r_ready()
except Exception as e:
    print(f"[gp_utils] Warning: R environment not ready ({e}). Some R-related functions may not work.")

from .preprocessing import str2numConverter
from .reducers import NoOpReducer, LassoReducer
from .models import RRBLUPModel, BayesBModel, BayesRRModel, EGBLUPModel
from .simCross import read_cross_func, map_snp_order_func, sim_cross_with_genos
from .evaluations import pear_metric, pear_scorer, spear_metric, spear_scorer, top_r_portion_hit_rate, report_metrics, compute_top_mean

__all__ = [
    "str2numConverter",
    "NoOpReducer", "LassoReducer",
    "RRBLUPModel", "BayesBModel", "BayesRRModel", "EGBLUPModel",
    "read_cross_func", "map_snp_order_func", "sim_cross_with_genos",
    "pear_metric", "pear_scorer", "spear_metric", "spear_scorer", "top_r_portion_hit_rate", "report_metrics", "compute_top_mean"
]

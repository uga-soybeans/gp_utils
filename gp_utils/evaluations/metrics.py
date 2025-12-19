import numpy as np
from scipy import stats
from sklearn.metrics import make_scorer


def pear_metric(a, b):
    '''
    a, b: array-like objects.
    '''
    return stats.pearsonr(a, b)[0]

pear_scorer = make_scorer(pear_metric)


def spear_metric(a, b):
    '''
    a, b: array-like objects.
    '''
    return stats.spearmanr(a, b)[0]

spear_scorer = make_scorer(spear_metric)


def top_r_portion_hit_rate(y_true, y_pred, r=0.25):
    '''
    y_true, y_pred: better be numpy arrays
    '''
    if not (0 <= r <= 1):
        raise ValueError("r must be in the interval [0, 1]")
    
    n = len(y_true)
    k = max(1, int(n * r))  # Ensure at least one element is considered
    
    ### Get indices of the top k elements
    top_true_indices = np.argsort(y_true)[-k:]
    top_pred_indices = np.argsort(y_pred)[-k:]
    
    ### Compute the hit rate
    hits = len(set(top_true_indices) & set(top_pred_indices))
    return hits / k


def report_metrics(y_true, y_pred, _r=0.25, rep=None, fold=None):
    '''
    y_true: lists 
    y_pred: lists
    _r: float between 0 and 1.
    _rep, fold: for CV metrics
    '''
    res = {}
    if rep and fold:
        res["rep"] = rep 
        res["fold"] = fold
    res["Pearson's r"] = pear_metric(y_true, y_pred)
    res[f"Top {int(_r * 100)}% HR"] = top_r_portion_hit_rate(y_true, y_pred, r=_r)
    res[f"Low {int(_r * 100)}% HR"] = top_r_portion_hit_rate(-y_true, -y_pred, r=_r)
    return res


def compute_top_mean(num_lst, r):
    '''
    r: [0, 1]
    Computes the mean of the top r portion of num_lst.
    '''
    n = len(num_lst)
    k = max(1, int(n * r))  # Ensure at least one element is considered
    return np.mean(sorted(num_lst)[-k:])

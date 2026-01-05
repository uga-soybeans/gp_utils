import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso


class NoOpReducer(BaseEstimator, TransformerMixin):
    '''
    X: numpy array or pandas dataframe
    y: numpy array or pandas series
    '''
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return np.copy(X)


class LassoReducer(BaseEstimator, TransformerMixin):
    def __init__(self, alpha=0.1, r=0.1, test_size=0.2, n_reps=200, max_iter=10000):
        self.alpha = alpha
        self.r = r
        self.test_size = test_size
        self.n_reps = n_reps
        self.max_iter = max_iter
        self.selects_ind = [] # lst[lst[bool]]

    def fit(self, X, y):
        '''
        X: numpy array or pandas dataframe
        y: numpy array or pandas series
        '''
        self.selects_ind = []
        for i in range(self.n_reps):
            X_tr, _, y_tr, _ = train_test_split(X, y, test_size=self.test_size, random_state=i)
            l_model = Lasso(alpha=self.alpha, max_iter=self.max_iter).fit(X_tr, y_tr)

            select = abs(l_model.coef_) > 0
            self.selects_ind.append(select)
        return self
    
    def transform(self, X):
        if not self.selects_ind:
            raise ValueError("The reducer has not been fitted yet.")
        agg_ind = self._aggregate_binary_lists(selects_ind=self.selects_ind, r=self.r)
        return X[:, agg_ind]

    ### Internal utilities ###
    def _aggregate_binary_lists(self, selects_ind, r):
        '''
        Filtered union of binary lists.

        Parameters
        ----------
        selects_ind: list of binary lists
        r: ratio
        '''
        ones_count = [sum(row[i] for row in selects_ind) for i in range(len(selects_ind[0]))] # Compute the number of 1s at each position
        threshold = r * len(selects_ind) # Determine the threshold
        if threshold > max(ones_count):
            return np.array(ones_count) > 0 # If threshold too high, return features that have been selected at least once.
        return np.array(ones_count) > threshold # Construct the output list
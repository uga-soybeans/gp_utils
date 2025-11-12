import numpy as np

import rpy2.robjects as robjects
from rpy2.robjects import default_converter
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import numpy2ri

from sklearn.base import BaseEstimator, RegressorMixin

################
### R models ###
################
class RRBLUPModel(BaseEstimator, RegressorMixin):
    def __init__(self):
        self.beta = None
        self.u = None
    
    def fit(self, X, y):
        '''
        X: numpy array or output of feature-engine.
        y: pandas series.
        '''
        if type(X) != np.ndarray:
            X = X.values
        robjects.r('''
            rrblup_fit <- function(X, y) {
                library(rrBLUP)
                model <- mixed.solve(y=as.numeric(y), Z=as.matrix(X))
                return(model)
            }
        ''')
        fit_func = robjects.r['rrblup_fit']
        with localconverter(default_converter + numpy2ri.converter):
            model = fit_func(X, y.values)
        with localconverter(default_converter + numpy2ri.converter):
            self.u = np.array(model.rx2('u'))
            self.beta = np.array(model.rx2('beta'))
        self.is_fitted_ = True
        return self
    
    def predict(self, X):
        '''
        X: numpy array or output of feature-engine.
        '''
        if (not self.is_fitted_) or (self.beta is None) or (self.u is None):
            raise ValueError("Model has not been trained.")
        if type(X) != np.ndarray:
            X = X.values
        return X @ self.u + self.beta


class BayesBModel(BaseEstimator, RegressorMixin):
    def __init__(self):
        self.beta = None
        self.u = None

    def fit(self, X, y):
        '''
        X: numpy array or output of feature-engine.
        y: pandas series.
        '''
        if type(X) != np.ndarray:
            X = X.values
        robjects.r('''
            bb_fit <- function(X, y) {
            library(bWGR)
            model <- wgr(y=as.numeric(y), iv=TRUE, pi=.95, X=as.matrix(X)+1)
            return(model)
            }
        ''')
        fit_func = robjects.r['bb_fit']
        with localconverter(default_converter + numpy2ri.converter):
            model = fit_func(X, y.values)
        with localconverter(default_converter + numpy2ri.converter):
            self.u = np.array(model.rx2("b"))
            self.beta = np.array(model.rx2("mu"))
        self.is_fitted_ = True
        return self
    
    def predict(self, X):
        '''
        X: numpy array or output of feature-engine.
        '''
        if (not self.is_fitted_) or (self.beta is None) or (self.u is None):
            raise ValueError("Model has not been trained.")
        if type(X) != np.ndarray:
            X = X.values
        return (X + 1) @ self.u + self.beta


class BayesRRModel(BaseEstimator, RegressorMixin):
    def __init__(self):
        self.beta = None
        self.u = None

    def fit(self, X, y):
        '''
        X: numpy array or output of feature-engine.
        y: pandas series.
        '''
        if type(X) != np.ndarray:
            X = X.values
        robjects.r('''
            brr_fit <- function(X, y) {
            library(bWGR)
            model <- wgr(y=as.numeric(y), iv=FALSE, pi=0, X=as.matrix(X)+1)
            return(model)
            }
        ''')
        fit_func = robjects.r['brr_fit']
        with localconverter(default_converter + numpy2ri.converter):
            model = fit_func(X, y.values)
        with localconverter(default_converter + numpy2ri.converter):
            self.u = np.array(model.rx2("b"))
            self.beta = np.array(model.rx2("mu"))
        self.is_fitted_ = True
        return self
    
    def predict(self, X):
        '''
        X: numpy array or output of feature-engine.
        '''
        if (not self.is_fitted_) or (self.beta is None) or (self.u is None):
            raise ValueError("Model has not been trained.")
        if type(X) != np.ndarray:
            X = X.values
        return (X + 1) @ self.u + self.beta


class EGBLUPModel(BaseEstimator, RegressorMixin):
    def __init__(self):
        self.genos = None
        self.phenos = None

    def fit(self, X, y):
        '''
        X: numpy array or output of feature-engine.
        y: pandas series.
        '''
        if type(X) != np.ndarray:
            X = X.values
        self.genos = X
        self.phenos = y.values
        self.is_fitted_ = True
        return self
    
    def predict(self, X):
        '''
        X: numpy array or output of feature-engine.
        '''
        if (not self.is_fitted_) or (self.genos is None) or (self.phenos is None):
            raise ValueError("Model has not been trained.")
        if type(X) != np.ndarray:
            X = X.values
        robjects.r('''
            egblup_fit <- function(bigX, bigy, train_length) {
            library(Matrix)
            library(rrBLUP)
            library(EMMREML)
            total_length <- nrow(bigX)
            test_indices <- c(rep(FALSE, train_length), rep(TRUE, total_length-train_length))

            kin <- A.mat(bigX)
            epi <- kin * kin
            
            model <- emmremlMultiKernel(y=as.numeric(bigy[!test_indices]),
                   X=matrix(rep(1,train_length), ncol=1),
                   Zlist=list(as.matrix(diag(total_length)[!test_indices,]), as.matrix(diag(total_length)[!test_indices,])),
                   Klist=list(as.matrix(kin), as.matrix(epi)))
            return(model)
            }
        ''')
        
        fit_func = robjects.r['egblup_fit']

        bigX, bigy = self.genos, self.phenos
        train_length = bigX.shape[0]
        train_flag = np.array_equal(X, bigX)

        if not train_flag:
            bigX = np.vstack((bigX, X))
            bigy = np.concatenate((bigy, np.full(X.shape[0], np.nan)))

        with localconverter(default_converter + numpy2ri.converter):
            model = fit_func(bigX, bigy, train_length)
        
        with localconverter(default_converter + numpy2ri.converter):
            uhat = np.array(model.rx2("uhat"))
            betahat = np.array(model.rx2("betahat"))
        
        muhat = uhat.reshape(-1, 2, order='F')
        total_pred =  muhat.sum(axis=1) + betahat.item()

        if not train_flag:
            return total_pred[train_length:]

        return total_pred

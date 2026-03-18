import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from feature_engine.selection import DropConstantFeatures

from preprocessing import str2numConverter
from reducers import init_reducer
from models import init_model
from evaluations import pear_scorer
# from ..preprocessing import str2numConverter
# from ..reducers import init_reducer
# from ..models import init_model
# from ..evaluations import pear_scorer

def init_pipeline(reducer_name: str, model_name: str, preprocess_params: dict, reducer_params: dict, model_params: dict, random_state: int = 42):
    '''
    Initialize a reducer + regressor pipeline with the given hyperparameters.

    preprocess_params: Preprocessing configuration.
    
    reducer_params: Hyperparameter setting for the specified reducer.
                    Different reducers have completely different hyperparameters.
    
    model_params: Hyperparameters setting for the specified algorithm.
                  Different algorithms have completely different hyperparameters.
    '''
    dropconstant = DropConstantFeatures(missing_values='ignore')
    str2num = str2numConverter()
    imp = SimpleImputer(missing_values=np.nan, strategy=preprocess_params['imputation-strategy'], fill_value=preprocess_params['imputation-fill-value'])
    scaler = StandardScaler()
    reducer_model = init_reducer(reducer_name=reducer_name, reducer_params=reducer_params, random_state=random_state)
    regressor_model = init_model(model_name=model_name, model_params=model_params, random_state=random_state)
    return Pipeline([
        ('dropconstant', dropconstant),
        ('converter', str2num),
        ('imputer', imp),
        ('scaler', scaler),
        (reducer_name, reducer_model),
        (model_name, regressor_model)
    ])


def train_pipeline(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        preprocess_params: dict,
        reducer_name: str,
        reducer_params: dict,
        model_name: str,
        model_params: dict,
        random_state: int = 42,
        reducer_param_grid: dict = None,
        model_param_grid: dict = None,
        gridsearch_cv_folds: int = 10,
        result_path: str = None,
        run_gridsearch = False,
):
    '''
    Train a reducer + regressor pipeline or perform gridsearch and record results.

    reducer_param_grid, model_param_grid: Dictionaries of the form {'RFR__n_estimators': [100, 200, 400], 'RFR__max_depth': [3, 4, 8]} if not None
                                          Warning: Unlike reducer initialization, parameters MUST match the original argument names in function or class definition.

    gridsearch_cv_folds: number of folds to use for gridsearch cross validation
    
    result_path: path to store gridsearch result (.csv)
    '''
    pipeline = init_pipeline(
        reducer_name=reducer_name,
        model_name=model_name,
        preprocess_params=preprocess_params,
        reducer_params=reducer_params,
        model_params=model_params,
        random_state=random_state
    )

    if run_gridsearch:
        assert reducer_param_grid is not None
        assert model_param_grid is not None

        gridsearchcv = GridSearchCV(
            estimator=pipeline,
            param_grid=(reducer_param_grid | model_param_grid), # Union of two dictionaries
            cv=gridsearch_cv_folds,
            scoring=pear_scorer,
            verbose=3,
            refit=True
        )
        gridsearchcv.fit(X_train, y_train)
    
        result_path = result_path if result_path is not None else f"{model_name}_{reducer_name}_gridsearch.csv"
        pd.DataFrame(gridsearchcv.cv_results_).to_csv(result_path, index=False)
    
        return gridsearchcv.best_estimator_
    
    else:
        pipeline.fit(X_train, y_train)
        return pipeline

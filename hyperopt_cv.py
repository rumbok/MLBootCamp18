from hyperopt import STATUS_OK
from __future__ import unicode_literals
import gc
from time import time
from skopt import BayesSearchCV
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier, Dataset
from sklearn.model_selection import PredefinedSplit, RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os

from data_loading import load_csi_test, load_csi_train, load_features
from data_prepare import merge_features
from transformers.pandas_subset import PandasSubset

os.environ['MKL_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'

N_FOLDS = 10


if __name__ == '__main__':
    train_df = load_csi_train()
    train_feat_df = load_features('train')

    train_df = merge_features(train_df, train_feat_df)
    train_y = train_df['CSI']
    train_X = train_df.drop(['CSI', 'CONTACT_DATE', 'SNAP_DATE'], axis=1)
    gc.collect()

    train_set = Dataset(train_X, train_y)


def objective(params, n_folds=N_FOLDS):
    cv_results = lgb.cv(params, train_set, nfold=n_folds, num_boost_round=10000,
                        early_stopping_rounds=100, metrics='auc', seed=50)

    best_score = max(cv_results['auc-mean'])

    loss = 1 - best_score

    return {'loss': loss, 'params': params, 'status': STATUS_OK}
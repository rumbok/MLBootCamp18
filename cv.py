from __future__ import unicode_literals
import gc
from time import time
from skopt import BayesSearchCV
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.model_selection import PredefinedSplit, RepeatedStratifiedKFold
import os

from data_loading import load_csi_test, load_csi_train, load_features
from data_prepare import merge_features
from transformers.pandas_subset import PandasSubset

os.environ['MKL_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'

features = [
    'app',
    # 'device',
    'os',
    'channel',
    'hour',
    # 'id',

    # 'id_app',
    # 'id_channel',
    # 'app_channel',
    # 'os_channel',
    'app_device',
    'ip_app',
    # 'app_os',

    'ip_count',
    'id_count',
    'channel_count',
    'app_count',
    'device_count',
    'os_count',

    'ip_app_count',
    'ip_channel_count',
    # 'ip_app_os_count',
    'ip_device_count',
    'app_channel_count',
    'device_os_count',

    'id_app_count',
    'id_channel_count',

    # 'ip_device_os_app_channel_count',

    'channel_app_nunique',
    # 'app_channel_nunique',
    'ip_channel_nunique',
    'ip_app_nunique',
    'ip_device_nunique',
    'device_os_nunique',
    # 'id_channel_nunique',
    'id_app_nunique',

    'channel_app_unique_ratio',
    'app_channel_unique_ratio',
    # 'ip_channel_unique_ratio',
    'ip_app_unique_ratio',
    'ip_device_unique_ratio',
    'device_os_unique_ratio',
    # 'id_channel_unique_ratio',
    # 'id_app_unique_ratio',

    'ip_app_ratio',
    # 'ip_channel_ratio',
    'app_channel_ratio',
    'device_os_ratio',
    'id_app_ratio',
    'id_channel_ratio',

    'ip_day_hour_count',
    'app_day_hour_count',
    'channel_day_hour_count',
    'ip_app_day_hour_count',
    'ip_app_os_day_hour_count',
    'ip_device_day_hour_count',
    # 'app_channel_day_hour_count',

    'id_day_hour_count',
    # 'id_app_day_hour_count',
    'id_channel_day_hour_count',

    'ip_duplicates_mean',
    # 'device_duplicates_mean',
    'os_duplicates_mean',
    'app_duplicates_mean',
    'channel_duplicates_mean',

    'ip_after_first_days',
    'id_after_first_days',
    'app_after_first_days',

    # 'ip_prev_click',
    'id_prev_click',

    'ip_prev_click_mean',
    'id_prev_click_mean',

    'next_click',
]
categorical = ['app',
               # 'device',
               'os', 'channel',
               # 'id',
               # 'id_app', 'id_channel',
               # 'app_channel', 'os_channel',
               'app_device', 'ip_app',
               # 'app_os'
               ]
target = 'is_attributed'


def status_print(optim_result):
    """Status callback during bayesian hyperparameter search"""
    all_models = pd.DataFrame(bayes_cv_tuner.cv_results_)

    best_params = pd.Series(bayes_cv_tuner.best_params_)
    print('Model #{}\nBest ROC-AUC: {}\nBest params: {}\n'.format(
        len(all_models),
        np.round(bayes_cv_tuner.best_score_, 4),
        bayes_cv_tuner.best_params_
    ))

    # Save all model results
    clf_name = bayes_cv_tuner.estimator.__class__.__name__
    all_models.to_csv(clf_name + "_lgb_cv_results.csv")


class FeaturePredictor(BaseEstimator):
    def __init__(self, **params):
        self.pipeline = Pipeline([
            ('subset', PandasSubset(**{k: True for k in features})),
            ('lgb', LGBMClassifier(objective='binary',
                                   learning_rate=0.2,
                                   num_leaves=7,
                                   max_depth=-1,
                                   min_child_samples=100,
                                   max_bin=105,
                                   subsample=0.7,
                                   subsample_freq=1,
                                   colsample_bytree=0.8,
                                   min_child_weight=0,
                                   subsample_for_bin=200000,
                                   min_split_gain=0,
                                   reg_alpha=0,
                                   reg_lambda=0,
                                   n_estimators=500,
                                   n_jobs=2,
                                   scale_pos_weight=99,
                                   )),
        ])
        self.set_params(**params)

    def fit(self, X, y, **fit_params):
        Xs = self.pipeline.named_steps['subset'].fit_transform(X)

        feats = self.pipeline.named_steps['subset'].fields()
        print(feats)
        cats = list(set(fit_params['categorical']).intersection(feats))
        print(cats)
        Xs_eval = self.pipeline.named_steps['subset'].transform(fit_params['X_eval'])
        y_eval = fit_params['y_eval']
        self.pipeline.named_steps['lgb'].fit(Xs, y,
                                             eval_metric="auc",
                                             eval_set=(Xs_eval, y_eval),
                                             early_stopping_rounds=30,
                                             verbose=10,
                                             feature_name=feats,
                                             categorical_feature=cats,
                                             )
        del Xs, y, Xs_eval, y_eval, feats, cats
        gc.collect()
        return self

    def predict(self, X):
        Xs = self.pipeline.named_steps['subset'].transform(X)
        return self.pipeline.named_steps['lgb'].predict_proba(Xs)[:, 1]

    def predict_proba(self, X):
        Xs = self.pipeline.named_steps['subset'].transform(X)
        return self.pipeline.named_steps['lgb'].predict_proba(Xs)

    def get_params(self, deep=True):
        return self.pipeline.get_params(deep)

    def set_params(self, **params):
        self.pipeline.set_params(**params)
        return self


if __name__ == '__main__':
    train_df, val_df = load_data(TRAIN_DAYS, VALID_DAY)

    print("train size: ", len(train_df))
    print("val size : ", len(val_df))

    len_train = len(train_df)
    len_valid = len(val_df)
    train_df = train_df.append(val_df)
    gc.collect()

    print("Training...")
    start_time = time()

    search_spaces = {
        'subset__app': [True, False],
        # 'subset__device': [True, False],
        'subset__os': [True, False],
        'subset__channel': [True, False],
        'subset__hour': [True, False],
        # 'subset__id': [True, False],

        # 'subset__id_app': [True, False],
        # 'subset__id_channel': [True, False],
        # 'subset__app_channel': [True, False],
        # 'subset__os_channel': [True, False],
        'subset__app_device': [True, False],
        'subset__ip_app': [True, False],
        # 'subset__app_os': [True, False],

        'subset__ip_count': [True, False],
        'subset__id_count': [True, False],
        'subset__channel_count': [True, False],
        'subset__app_count': [True, False],
        'subset__device_count': [True, False],
        'subset__os_count': [True, False],

        'subset__ip_app_count': [True, False],
        'subset__ip_channel_count': [True, False],
        # 'subset__ip_app_os_count': [True, False],
        'subset__ip_device_count': [True, False],
        'subset__app_channel_count': [True, False],
        'subset__device_os_count': [True, False],

        'subset__id_app_count': [True, False],
        'subset__id_channel_count': [True, False],

        # 'subset__ip_device_os_app_channel_count': [True, False],

        'subset__channel_app_nunique': [True, False],
        # 'subset__app_channel_nunique': [True, False],
        'subset__ip_channel_nunique': [True, False],
        'subset__ip_app_nunique': [True, False],
        'subset__ip_device_nunique': [True, False],
        'subset__device_os_nunique': [True, False],
        # 'subset__id_channel_nunique': [True, False],
        'subset__id_app_nunique': [True, False],

        'subset__channel_app_unique_ratio': [True, False],
        'subset__app_channel_unique_ratio': [True, False],
        # 'subset__ip_channel_unique_ratio': [True, False],
        'subset__ip_app_unique_ratio': [True, False],
        'subset__ip_device_unique_ratio': [True, False],
        'subset__device_os_unique_ratio': [True, False],
        # 'subset__id_channel_unique_ratio': [True, False],
        # 'subset__id_app_unique_ratio': [True, False],

        'subset__ip_app_ratio': [True, False],
        # 'subset__ip_channel_ratio': [True, False],
        'subset__app_channel_ratio': [True, False],
        'subset__device_os_ratio': [True, False],
        'subset__id_app_ratio': [True, False],
        'subset__id_channel_ratio': [True, False],

        'subset__ip_day_hour_count': [True, False],
        'subset__app_day_hour_count': [True, False],
        'subset__channel_day_hour_count': [True, False],
        'subset__ip_app_day_hour_count': [True, False],
        'subset__ip_app_os_day_hour_count': [True, False],
        'subset__ip_device_day_hour_count': [True, False],
        # 'subset__app_channel_day_hour_count': [True, False],

        'subset__id_day_hour_count': [True, False],
        # 'subset__id_app_day_hour_count': [True, False],
        'subset__id_channel_day_hour_count': [True, False],

        'subset__ip_duplicates_mean': [True, False],
        # 'subset__device_duplicates_mean': [True, False],
        'subset__os_duplicates_mean': [True, False],
        'subset__app_duplicates_mean': [True, False],
        'subset__channel_duplicates_mean': [True, False],

        'subset__ip_after_first_days': [True, False],
        'subset__id_after_first_days': [True, False],
        'subset__app_after_first_days': [True, False],

        'subset__ip_prev_click': [True, False],
        'subset__id_prev_click': [True, False],

        'subset__ip_prev_click_mean': [True, False],
        'subset__id_prev_click_mean': [True, False],

        'subset__next_click': [True, False],

        'lgb__num_leaves': (3, 21),
        # 'lgb__max_depth': (2, 50),
        'lgb__min_child_samples': (1, 500),
        'lgb__max_bin': (10, 2000),
        'lgb__subsample': (0.1, 1.0, 'uniform'),
        # 'lgb__subsample_freq': (0, 10),
        'lgb__colsample_bytree': (0.5, 1.0, 'uniform'),
        'lgb__min_child_weight': (0, 50),
        # 'lgb__subsample_for_bin': (100000, 500000),
        # 'lgb__reg_lambda': (1e-9, 1000, 'log-uniform'),
        # 'lgb__reg_alpha': (1e-9, 1.0, 'log-uniform'),
        'lgb__scale_pos_weight': (1, 300, 'log-uniform'),
    }

    test_fold = np.zeros(len(train_df))
    test_fold[:len_train] = -1

    bayes_cv_tuner = BayesSearchCV(
        estimator=FeaturePredictor(),
        search_spaces=search_spaces,
        scoring='roc_auc',
        cv=RepeatedStratifiedKFold(test_fold),
        n_jobs=1,
        pre_dispatch=1,
        n_iter=30,
        verbose=0,
        refit=False,
        random_state=42,
        fit_params={'categorical': categorical, 'X_eval': val_df, 'y_eval': val_df[target].values}
    )

    result = bayes_cv_tuner.fit(train_df, train_df[target].values, callback=status_print)

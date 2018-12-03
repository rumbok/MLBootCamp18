from __future__ import unicode_literals
import gc
from time import time
from skopt import BayesSearchCV
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline, FeatureUnion
from lightgbm import LGBMClassifier, Dataset
from sklearn.model_selection import PredefinedSplit, RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.svm import SVC
import os

from data_loading import load_csi_test, load_csi_train, load_features
from data_prepare import merge_features
from transformers.pandas_select import PandasSelect
from transformers.pandas_subset import PandasSubset

os.environ['MKL_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'

features = [
    'COM_CAT#1', 'COM_CAT#2', 'COM_CAT#3',
    'BASE_TYPE', 'ACT', 'ARPU_GROUP', 'COM_CAT#7', 'COM_CAT#8',
    'DEVICE_TYPE_ID', 'INTERNET_TYPE_ID', 'REVENUE', 'ITC', 'VAS',
    'RENT_CHANNEL', 'ROAM', 'COST', 'COM_CAT#17', 'COM_CAT#18',
    'COM_CAT#19', 'COM_CAT#20', 'COM_CAT#21', 'COM_CAT#22', 'COM_CAT#23',
    # 'COM_CAT#24',
    'COM_CAT#25', 'COM_CAT#26', 'COM_CAT#27', 'COM_CAT#28',
    'COM_CAT#29', 'COM_CAT#30', 'COM_CAT#31', 'COM_CAT#32', 'COM_CAT#33',
    'COM_CAT#34'
]
categorical = [
    'COM_CAT#1',
    'COM_CAT#2',
    'COM_CAT#3',
    'BASE_TYPE',
    'ACT',
    'ARPU_GROUP',
    'COM_CAT#7',
    'COM_CAT#8',
    'DEVICE_TYPE_ID',
    'INTERNET_TYPE_ID',
    'COM_CAT#25', 'COM_CAT#26',
    'COM_CAT#34'
]
target = 'CSI'


def status_print(optim_result):
    """Status callback during bayesian hyperparameter search"""
    all_models = pd.DataFrame(bayes_cv_tuner.cv_results_)

    best_params = pd.Series(bayes_cv_tuner.best_params_)
    print(f'Best ROC-AUC: {np.round(bayes_cv_tuner.best_score_, 4),}')
    # print('Model #{}\nBest ROC-AUC: {}\nBest params: {}\n'.format(
    #     len(all_models),
    #     np.round(bayes_cv_tuner.best_score_, 4),
    #     bayes_cv_tuner.best_params_
    # ))

    # Save all model results
    clf_name = bayes_cv_tuner.estimator.__class__.__name__
    all_models.to_csv(clf_name + "_lgb_cv_results.csv")


if __name__ == '__main__':
    train_df = load_csi_train()
    train_feat_df = load_features('train')

    train_df = merge_features(train_df, train_feat_df)
    train_y = train_df['CSI']
    train_X = train_df.drop(['CSI', 'CONTACT_DATE', 'SNAP_DATE'], axis=1)
    gc.collect()


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
                                       n_jobs=4,
                                       is_unbalance=True,
                                       random_state=42,
                                       class_weight='balanced'
                                       )),
            ])
            self.set_params(**params)

        def fit(self, X, y, **fit_params):
            Xs = self.pipeline.named_steps['subset'].fit_transform(train_X.loc[X])

            feats = self.pipeline.named_steps['subset'].fields()
            cats = list(set(categorical).intersection(feats))
            Xs_eval = self.pipeline.named_steps['subset'].transform(train_X.loc[train_X.index.difference(X)])
            y_eval = train_y.loc[train_X.index.difference(X)]
            self.pipeline.named_steps['lgb'].fit(Xs, y,
                                                 eval_metric="auc",
                                                 eval_set=(Xs_eval, y_eval),
                                                 early_stopping_rounds=100,
                                                 verbose=1,
                                                 feature_name=feats,
                                                 categorical_feature=cats,
                                                 )
            del Xs, y, Xs_eval, y_eval, feats, cats
            gc.collect()
            return self

        def predict(self, X):
            Xs = self.pipeline.named_steps['subset'].transform(train_X.loc[X])
            return self.pipeline.named_steps['lgb'].predict_proba(Xs)[:, 1]

        def predict_proba(self, X):
            Xs = self.pipeline.named_steps['subset'].transform(train_X.loc[X])
            return self.pipeline.named_steps['lgb'].predict_proba(Xs)

        def get_params(self, deep=True):
            return self.pipeline.get_params(deep)

        def set_params(self, **params):
            self.pipeline.set_params(**params)
            return self


    print("Training...")
    start_time = time()

    search_spaces = {
        'subset__COM_CAT#1': [True, False],
        'subset__COM_CAT#2': [True, False],
        'subset__COM_CAT#3': [True, False],
        'subset__BASE_TYPE': [True, False],
        'subset__ACT': [True, False],
        'subset__ARPU_GROUP': [True, False],
        'subset__COM_CAT#7': [True, False],
        'subset__COM_CAT#8': [True, False],
        'subset__DEVICE_TYPE_ID': [True, False],
        'subset__INTERNET_TYPE_ID': [True, False],
        'subset__REVENUE': [True, False],
        'subset__ITC': [True, False],
        'subset__VAS': [True, False],
        'subset__RENT_CHANNEL': [True, False],
        'subset__ROAM': [True, False],
        'subset__COST': [True, False],
        'subset__COM_CAT#17': [True, False],
        'subset__COM_CAT#18': [True, False],
        'subset__COM_CAT#19': [True, False],
        'subset__COM_CAT#20': [True, False],
        'subset__COM_CAT#21': [True, False],
        'subset__COM_CAT#22': [True, False],
        'subset__COM_CAT#23': [True, False],
        'subset__COM_CAT#25': [True, False],
        'subset__COM_CAT#26': [True, False],
        'subset__COM_CAT#27': [True, False],
        'subset__COM_CAT#28': [True, False],
        'subset__COM_CAT#29': [True, False],
        'subset__COM_CAT#30': [True, False],
        'subset__COM_CAT#31': [True, False],
        'subset__COM_CAT#32': [True, False],
        'subset__COM_CAT#33': [True, False],
        'subset__COM_CAT#34': [True, False],

        # 'lgb__num_leaves': (3, 21),
        # # 'lgb__n_estimators': (1, 100),
        # 'lgb__max_depth': (2, 50),
        # 'lgb__min_child_samples': (1, 500),
        # 'lgb__max_bin': (10, 2000),
        # 'lgb__subsample': (0.1, 1.0, 'uniform'),
        # 'lgb__subsample_freq': (0, 10),
        # 'lgb__colsample_bytree': (0.5, 1.0, 'uniform'),
        # 'lgb__min_child_weight': (0, 50),
        # 'lgb__subsample_for_bin': (100000, 500000),
        # 'lgb__reg_lambda': (1e-9, 1000, 'log-uniform'),
        # 'lgb__reg_alpha': (1e-9, 1.0, 'log-uniform'),
    }
    ppl = Pipeline([
            ('subset', PandasSubset(**{k: True for k in features})),
            ('vectorizer', FeatureUnion([
                ('REVENUE', PandasSelect('REVENUE', fillna_zero=True)),
                ('ITC', PandasSelect('ITC', fillna_zero=True)),
                ('VAS', PandasSelect('VAS', fillna_zero=True)),
                ('RENT_CHANNEL', PandasSelect('RENT_CHANNEL', fillna_zero=True)),
                ('ROAM', PandasSelect('ROAM', fillna_zero=True)),
                ('COST', PandasSelect('COST', fillna_zero=True)),
                ('COM_CAT#17', PandasSelect('COM_CAT#17', fillna_zero=True)),
                ('COM_CAT#18', PandasSelect('COM_CAT#18', fillna_zero=True)),
                ('COM_CAT#19', PandasSelect('COM_CAT#19', fillna_zero=True)),
                ('COM_CAT#20', PandasSelect('COM_CAT#20', fillna_zero=True)),
                ('COM_CAT#21', PandasSelect('COM_CAT#21', fillna_zero=True)),
                ('COM_CAT#22', PandasSelect('COM_CAT#22', fillna_zero=True)),
                ('COM_CAT#23', PandasSelect('COM_CAT#23', fillna_zero=True)),
                ('COM_CAT#27', PandasSelect('COM_CAT#27', fillna_zero=True)),
                ('COM_CAT#28', PandasSelect('COM_CAT#28', fillna_zero=True)),
                ('COM_CAT#29', PandasSelect('COM_CAT#29', fillna_zero=True)),
                ('COM_CAT#30', PandasSelect('COM_CAT#30', fillna_zero=True)),
                ('COM_CAT#31', PandasSelect('COM_CAT#31', fillna_zero=True)),
                ('COM_CAT#32', PandasSelect('COM_CAT#32', fillna_zero=True)),
                ('COM_CAT#33', PandasSelect('COM_CAT#33', fillna_zero=True)),

                ('COM_CAT#1', Pipeline([
                    ('select', PandasSelect('COM_CAT#1', fillna_zero=True)),
                    ('ohe', OneHotEncoder(handle_unknown='ignore'))
                ])),
                ('COM_CAT#2', Pipeline([
                    ('select', PandasSelect('COM_CAT#2', fillna_zero=True)),
                    ('ohe', OneHotEncoder(handle_unknown='ignore'))
                ])),
                ('COM_CAT#3', Pipeline([
                    ('select', PandasSelect('COM_CAT#3', fillna_zero=True)),
                    ('ohe', OneHotEncoder(handle_unknown='ignore'))
                ])),
                ('BASE_TYPE', Pipeline([
                    ('select', PandasSelect('BASE_TYPE', fillna_zero=True)),
                    ('ohe', OneHotEncoder(handle_unknown='ignore'))
                ])),
                ('ACT', Pipeline([
                    ('select', PandasSelect('ACT', fillna_zero=True)),
                    ('ohe', OneHotEncoder(handle_unknown='ignore'))
                ])),
                ('ARPU_GROUP', Pipeline([
                    ('select', PandasSelect('ARPU_GROUP', fillna_zero=True)),
                    ('ohe', OneHotEncoder(handle_unknown='ignore'))
                ])),
                ('COM_CAT#7', Pipeline([
                    ('select', PandasSelect('COM_CAT#7', fillna_zero=True)),
                    ('ohe', OneHotEncoder(handle_unknown='ignore'))
                ])),
                ('COM_CAT#8', Pipeline([
                    ('select', PandasSelect('COM_CAT#8', fillna_zero=True)),
                    ('ohe', OneHotEncoder(handle_unknown='ignore'))
                ])),
                ('DEVICE_TYPE_ID', Pipeline([
                    ('select', PandasSelect('DEVICE_TYPE_ID', fillna_zero=True)),
                    ('ohe', OneHotEncoder(handle_unknown='ignore'))
                ])),
                ('INTERNET_TYPE_ID', Pipeline([
                    ('select', PandasSelect('INTERNET_TYPE_ID', fillna_zero=True)),
                    ('ohe', OneHotEncoder(handle_unknown='ignore'))
                ])),
                ('COM_CAT#25', Pipeline([
                    ('select', PandasSelect('COM_CAT#25', fillna_zero=True)),
                    ('ohe', OneHotEncoder(handle_unknown='ignore'))
                ])),
                ('COM_CAT#26', Pipeline([
                    ('select', PandasSelect('COM_CAT#26', fillna_zero=True)),
                    ('ohe', OneHotEncoder(handle_unknown='ignore'))
                ])),
                ('COM_CAT#34', Pipeline([
                    ('select', PandasSelect('COM_CAT#34', fillna_zero=True)),
                    ('ohe', OneHotEncoder(handle_unknown='ignore'))
                ])),
            ])),
            # ('logreg', LogisticRegression(random_state=42, class_weight='balanced', solver='liblinear', max_iter=1000))
            ('svm', SVC(gamma='auto', kernel='rbf', random_state=42, class_weight='balanced'))
    ])

    bayes_cv_tuner = BayesSearchCV(
        estimator=ppl,
        search_spaces=search_spaces,
        scoring='roc_auc',
        cv=RepeatedStratifiedKFold(5, 10, random_state=42),
        n_jobs=1,
        pre_dispatch=6,
        n_iter=50,
        verbose=0,
        refit=True,
        random_state=42,
    )

    result = bayes_cv_tuner.fit(train_X, train_y, callback=status_print)

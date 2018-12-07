from __future__ import unicode_literals
import gc
from skopt import BayesSearchCV
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline, FeatureUnion
from lightgbm import LGBMClassifier, Dataset
from sklearn.model_selection import PredefinedSplit, RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.naive_bayes import GaussianNB, ComplementNB
from sklearn.svm import SVC
import os
from datetime import datetime

from data_loading import load_csi_test, load_csi_train, load_features
from data_prepare import merge_features, add_weekday, add_holidays
from transformers.pandas_select import PandasSelect
from transformers.pandas_subset import PandasSubset

os.environ['MKL_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'

features = [
    'CONTACT_DATE_WEEKDAY',

    'CONTACT_DATE_0_HOLIDAYS',
    'CONTACT_DATE_1_HOLIDAYS',
    'CONTACT_DATE_2_HOLIDAYS',
    'CONTACT_DATE_3_HOLIDAYS',
]
categorical = [
    'CONTACT_DATE_WEEKDAY',

    'CONTACT_DATE_0_HOLIDAYS',
    'CONTACT_DATE_1_HOLIDAYS',
    'CONTACT_DATE_2_HOLIDAYS',
    'CONTACT_DATE_3_HOLIDAYS',
]
target = 'CSI'


def status_print(optim_result):
    """Status callback during bayesian hyperparameter search"""
    all_models = pd.DataFrame(bayes_cv_tuner.cv_results_)

    best_params = pd.Series(bayes_cv_tuner.best_params_)
    print(f'Best ROC-AUC: {np.round(bayes_cv_tuner.best_score_, 4),}, '
          f'current={np.round(bayes_cv_tuner.cv_results_["mean_test_score"][-1], 4)}, '
          f'std={np.round(bayes_cv_tuner.cv_results_["std_test_score"][-1], 4)}')
    # print('Model #{}\nBest ROC-AUC: {}\nBest params: {}\n'.format(
    #     len(all_models),
    #     np.round(bayes_cv_tuner.best_score_, 4),
    #     bayes_cv_tuner.best_params_
    # ))

    # Save all model results
    # if len(all_models)%10 == 0:
    #     clf_name = bayes_cv_tuner.estimator.named_steps['estimator'].__class__.__name__
    #     all_models.to_csv(f"cv_results/{clf_name}_cv_{datetime.now().strftime('%d_%H_%M')}.csv")


if __name__ == '__main__':
    train_df = load_csi_train()
    train_feat_df = load_features('train')

    train_df = merge_features(train_df, train_feat_df)
    train_df = add_weekday(train_df, 'CONTACT_DATE')
    train_df = add_holidays(train_df, 'CONTACT_DATE')
    print(train_df.head())
    train_y = train_df['CSI']
    train_X = train_df.drop(['CSI', 'CONTACT_DATE', 'SNAP_DATE'], axis=1)
    gc.collect()


    print("Training...")

    search_spaces = {
        'subset__CONTACT_DATE_WEEKDAY': [True, False],

        'subset__CONTACT_DATE_0_HOLIDAYS': [True, False],
        'subset__CONTACT_DATE_1_HOLIDAYS': [True, False],
        'subset__CONTACT_DATE_2_HOLIDAYS': [True, False],
        'subset__CONTACT_DATE_3_HOLIDAYS': [True, False],

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
                ('CONTACT_DATE_WEEKDAY', Pipeline([
                    ('select', PandasSelect('CONTACT_DATE_WEEKDAY', fillna_zero=True)),
                    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))
                ])),

                ('CONTACT_DATE_0_HOLIDAYS', Pipeline([
                    ('select', PandasSelect('CONTACT_DATE_0_HOLIDAYS', fillna_zero=True)),
                    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))
                ])),
                ('CONTACT_DATE_1_HOLIDAYS', Pipeline([
                    ('select', PandasSelect('CONTACT_DATE_1_HOLIDAYS', fillna_zero=True)),
                    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))
                ])),
                ('CONTACT_DATE_2_HOLIDAYS', Pipeline([
                    ('select', PandasSelect('CONTACT_DATE_2_HOLIDAYS', fillna_zero=True)),
                    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))
                ])),
                ('CONTACT_DATE_3_HOLIDAYS', Pipeline([
                    ('select', PandasSelect('CONTACT_DATE_2_HOLIDAYS', fillna_zero=True)),
                    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))
                ])),
            ])),
            ('estimator', LogisticRegression(random_state=42,
                                             penalty='l2',
                                             C=0.1,
                                             class_weight='balanced',
                                             solver='liblinear',
                                             max_iter=1000,
                                             n_jobs=1))
            # ('estimator', SVC(gamma='auto', kernel='rbf', random_state=42, class_weight='balanced'))
            # ('estimator', ComplementNB()),

    ])

    bayes_cv_tuner = BayesSearchCV(
        estimator=ppl,
        search_spaces=search_spaces,
        scoring='roc_auc',
        cv=RepeatedStratifiedKFold(4, 10),
        n_jobs=4,
        pre_dispatch=6,
        n_iter=100,
        verbose=0,
        refit=True,
        random_state=42,
    )

    result = bayes_cv_tuner.fit(train_X, train_y, callback=status_print)
    clf_name = bayes_cv_tuner.estimator.named_steps['estimator'].__class__.__name__
    all_models = pd.DataFrame(bayes_cv_tuner.cv_results_)
    all_models.to_csv(f"cv_results/{clf_name}_cv_{datetime.now().strftime('%d_%H_%M')}.csv")

    print(bayes_cv_tuner.best_estimator_.named_steps['estimator'].intercept_)
    print(bayes_cv_tuner.best_estimator_.named_steps['estimator'].coef_)

    test_df = load_csi_test()
    test_feat_df = load_features('test')

    test_df = merge_features(test_df, test_feat_df)
    test_df = add_weekday(test_df, 'CONTACT_DATE')
    test_df = add_holidays(test_df, 'CONTACT_DATE')
    test_X = test_df.drop(['CONTACT_DATE', 'SNAP_DATE'], axis=1)

    test_y = bayes_cv_tuner.predict_proba(test_X)

    df = pd.DataFrame(test_y[:, 1])
    df.to_csv(f"submits/date_"
              f"{bayes_cv_tuner.estimator.named_steps['estimator'].__class__.__name__}"
              f"_{datetime.now().strftime('%d_%H_%M')}"
              f"_{np.round(bayes_cv_tuner.best_score_, 4)}.csv",
              header=None,
              index=None)


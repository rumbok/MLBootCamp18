import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import gc
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from data_loading import load_csi_train, load_features, load_csi_test
from data_prepare import add_features, add_weekday, add_holidays, features, categorical
from transformers.pandas_select import PandasSelect
from transformers.pandas_subset import PandasSubset

ppl = Pipeline([
    ('subset', PandasSubset(**{k: True for k in features})),
    ('vectorizer', FeatureUnion([
        ('non-categorical', PandasSubset(**{k: True for k in features if k not in categorical})),

        ('COM_CAT#1', Pipeline([
            ('select', PandasSelect('COM_CAT#1', fillna_zero=True)),
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))
        ])),
        ('COM_CAT#2', Pipeline([
            ('select', PandasSelect('COM_CAT#2', fillna_zero=True)),
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))
        ])),
        ('COM_CAT#3', Pipeline([
            ('select', PandasSelect('COM_CAT#3', fillna_zero=True)),
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))
        ])),
        ('BASE_TYPE', Pipeline([
            ('select', PandasSelect('BASE_TYPE', fillna_zero=True)),
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))
        ])),
        ('ACT', Pipeline([
            ('select', PandasSelect('ACT', fillna_zero=True)),
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))
        ])),
        ('ARPU_GROUP', Pipeline([
            ('select', PandasSelect('ARPU_GROUP', fillna_zero=True)),
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))
        ])),
        ('COM_CAT#7', Pipeline([
            ('select', PandasSelect('COM_CAT#7', fillna_zero=True)),
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))
        ])),
        ('COM_CAT#8', Pipeline([
            ('select', PandasSelect('COM_CAT#8', fillna_zero=True)),
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))
        ])),
        ('DEVICE_TYPE_ID', Pipeline([
            ('select', PandasSelect('DEVICE_TYPE_ID', fillna_zero=True)),
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))
        ])),
        ('INTERNET_TYPE_ID', Pipeline([
            ('select', PandasSelect('INTERNET_TYPE_ID', fillna_zero=True)),
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))
        ])),
        ('COM_CAT#25', Pipeline([
            ('select', PandasSelect('COM_CAT#25', fillna_zero=True)),
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))
        ])),
        ('COM_CAT#26', Pipeline([
            ('select', PandasSelect('COM_CAT#26', fillna_zero=True)),
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))
        ])),
        ('COM_CAT#34', Pipeline([
            ('select', PandasSelect('COM_CAT#34', fillna_zero=True)),
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))
        ])),

        # ('CONTACT_DATE_WEEKDAY', Pipeline([
        #     ('select', PandasSelect('CONTACT_DATE_WEEKDAY', fillna_zero=True)),
        #     ('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))
        # ])),

        # ('CONTACT_DATE_0_HOLIDAYS', Pipeline([
        #     ('select', PandasSelect('CONTACT_DATE_0_HOLIDAYS', fillna_zero=True)),
        #     ('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))
        # ])),
        # ('CONTACT_DATE_1_HOLIDAYS', Pipeline([
        #     ('select', PandasSelect('CONTACT_DATE_1_HOLIDAYS', fillna_zero=True)),
        #     ('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))
        # ])),
        # ('CONTACT_DATE_2_HOLIDAYS', Pipeline([
        #     ('select', PandasSelect('CONTACT_DATE_2_HOLIDAYS', fillna_zero=True)),
        #     ('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))
        # ])),
        # ('CONTACT_DATE_3_HOLIDAYS', Pipeline([
        #     ('select', PandasSelect('CONTACT_DATE_2_HOLIDAYS', fillna_zero=True)),
        #     ('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))
        # ])),
    ])),
    ('estimator', RandomForestClassifier(n_estimators=100, n_jobs=4, random_state=42))
])


def adversial_train_test_split(train_X, train_y, test_X, topK=500):
    train_X['train'] = 1
    train_X['target'] = train_y
    test_X['train'] = 0
    test_X['target'] = -1

    df = pd.concat((train_X, test_X), sort=False).reset_index(drop=True)
    df[df.select_dtypes(include=[np.float16]).columns] = \
        df[df.select_dtypes(include=[np.float16]).columns].fillna(0.0)
    df[df.select_dtypes(include=[np.float32]).columns] = \
        df[df.select_dtypes(include=[np.float32]).columns].fillna(0.0)
    df[df.select_dtypes(include=[np.float64]).columns] = \
        df[df.select_dtypes(include=[np.float64]).columns].fillna(0.0)
    print(df.info())

    X = df.drop(['train', 'target'], axis=1)
    y = df['train']

    # tsne = TSNE(n_components=2, init='pca', verbose=1, random_state=42)
    # Y = tsne.fit_transform(X[features])
    # plt.scatter(x=Y[:, 0], y=Y[:, 1], c=y)
    # plt.title("t-SNE (train-test)")
    # plt.axis('tight')
    # plt.show()

    predictions = np.zeros(y.shape)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for f, (train_i, test_i) in enumerate(cv.split(X, y)):
        x_train = X.iloc[train_i]
        x_test = X.iloc[test_i]
        y_train = y.iloc[train_i]
        y_test = y.iloc[test_i]

        ppl.fit(x_train, y_train)

        p = ppl.predict_proba(x_test)[:, 1]

        auc = roc_auc_score(y_test, p)
        print(f'Train-test similarity AUC = {auc}')

        predictions[test_i] = p

    df['pred'] = predictions
    df = df[df['train'] == 1].sort_values(by=['pred'])
    test_df = df.head(topK)
    train_df = df.tail(len(df)-topK)
    return train_df.drop(['train', 'pred', 'target'], axis=1), \
           train_df['target'], \
           test_df.drop(['train', 'pred', 'target'], axis=1), \
           test_df['target']


if __name__ == '__main__':
    train_df = load_csi_train()
    train_feat_df = load_features('train')

    train_df = add_features(train_df, train_feat_df)
    train_df = add_weekday(train_df, 'CONTACT_DATE')
    train_df = add_holidays(train_df, 'CONTACT_DATE')
    train_X = train_df.drop(['CSI', 'CONTACT_DATE', 'SNAP_DATE'], axis=1)
    train_y = train_df['CSI']
    gc.collect()

    test_df = load_csi_test()
    test_feat_df = load_features('test')

    test_df = add_features(test_df, test_feat_df)
    test_df = add_weekday(test_df, 'CONTACT_DATE')
    test_df = add_holidays(test_df, 'CONTACT_DATE')
    test_X = test_df.drop(['CONTACT_DATE', 'SNAP_DATE'], axis=1)
    gc.collect()

    res_df = adversial_train_test_split(train_X, train_y, test_X, 1000)
    # print(res_df)

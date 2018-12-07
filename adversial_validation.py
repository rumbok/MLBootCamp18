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
from data_prepare import merge_features, add_weekday, add_holidays, features
from transformers.pandas_select import PandasSelect
from transformers.pandas_subset import PandasSubset

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
        # ('COM_CAT#23', PandasSelect('COM_CAT#23', fillna_zero=True)),
        ('COM_CAT#27', PandasSelect('COM_CAT#27', fillna_zero=True)),
        ('COM_CAT#28', PandasSelect('COM_CAT#28', fillna_zero=True)),
        ('COM_CAT#29', PandasSelect('COM_CAT#29', fillna_zero=True)),
        # ('COM_CAT#30', PandasSelect('COM_CAT#30', fillna_zero=True)),
        ('COM_CAT#31', PandasSelect('COM_CAT#31', fillna_zero=True)),
        # ('COM_CAT#32', PandasSelect('COM_CAT#32', fillna_zero=True)),
        ('COM_CAT#33', PandasSelect('COM_CAT#33', fillna_zero=True)),

        ('COM_CAT#1', Pipeline([
            ('select', PandasSelect('COM_CAT#1', fillna_zero=True)),
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))
        ])),
        # ('COM_CAT#2', Pipeline([
        #     ('select', PandasSelect('COM_CAT#2', fillna_zero=True)),
        #     ('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))
        # ])),
        ('COM_CAT#3', Pipeline([
            ('select', PandasSelect('COM_CAT#3', fillna_zero=True)),
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))
        ])),
        ('BASE_TYPE', Pipeline([
            ('select', PandasSelect('BASE_TYPE', fillna_zero=True)),
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))
        ])),
        # ('ACT', Pipeline([
        #     ('select', PandasSelect('ACT', fillna_zero=True)),
        #     ('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))
        # ])),
        # ('ARPU_GROUP', Pipeline([
        #     ('select', PandasSelect('ARPU_GROUP', fillna_zero=True)),
        #     ('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))
        # ])),
        ('COM_CAT#7', Pipeline([
            ('select', PandasSelect('COM_CAT#7', fillna_zero=True)),
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))
        ])),
        # ('COM_CAT#8', Pipeline([
        #     ('select', PandasSelect('COM_CAT#8', fillna_zero=True)),
        #     ('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))
        # ])),
        ('DEVICE_TYPE_ID', Pipeline([
            ('select', PandasSelect('DEVICE_TYPE_ID', fillna_zero=True)),
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))
        ])),
        # ('INTERNET_TYPE_ID', Pipeline([
        #     ('select', PandasSelect('INTERNET_TYPE_ID', fillna_zero=True)),
        #     ('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))
        # ])),
        ('COM_CAT#25', Pipeline([
            ('select', PandasSelect('COM_CAT#25', fillna_zero=True)),
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))
        ])),
        ('COM_CAT#26', Pipeline([
            ('select', PandasSelect('COM_CAT#26', fillna_zero=True)),
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))
        ])),
        # ('COM_CAT#34', Pipeline([
        #     ('select', PandasSelect('COM_CAT#34', fillna_zero=True)),
        #     ('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))
        # ])),

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

        ('REVENUE_2m', PandasSelect('REVENUE_2m', fillna_zero=True)),
        ('ITC_2m', PandasSelect('ITC_2m', fillna_zero=True)),
        # ('VAS_2m', PandasSelect('VAS_2m', fillna_zero=True)),
        ('RENT_CHANNEL_2m', PandasSelect('RENT_CHANNEL_2m', fillna_zero=True)),
        ('ROAM_2m', PandasSelect('ROAM_2m', fillna_zero=True)),
        ('COST_2m', PandasSelect('COST_2m', fillna_zero=True)),
        # ('COM_CAT#17_2m', PandasSelect('COM_CAT#17_2m', fillna_zero=True)),
        ('COM_CAT#18_2m', PandasSelect('COM_CAT#18_2m', fillna_zero=True)),
        ('COM_CAT#19_2m', PandasSelect('COM_CAT#19_2m', fillna_zero=True)),
        ('COM_CAT#20_2m', PandasSelect('COM_CAT#20_2m', fillna_zero=True)),
        ('COM_CAT#21_2m', PandasSelect('COM_CAT#21_2m', fillna_zero=True)),
        ('COM_CAT#22_2m', PandasSelect('COM_CAT#22_2m', fillna_zero=True)),
        ('COM_CAT#23_2m', PandasSelect('COM_CAT#23_2m', fillna_zero=True)),
        ('COM_CAT#27_2m', PandasSelect('COM_CAT#27_2m', fillna_zero=True)),
        ('COM_CAT#28_2m', PandasSelect('COM_CAT#28_2m', fillna_zero=True)),
        ('COM_CAT#29_2m', PandasSelect('COM_CAT#29_2m', fillna_zero=True)),
        ('COM_CAT#30_2m', PandasSelect('COM_CAT#30_2m', fillna_zero=True)),
        ('COM_CAT#31_2m', PandasSelect('COM_CAT#31_2m', fillna_zero=True)),
        # ('COM_CAT#32_2m', PandasSelect('COM_CAT#32_2m', fillna_zero=True)),
        ('COM_CAT#33_2m', PandasSelect('COM_CAT#33_2m', fillna_zero=True)),

        ('REVENUE_3m', PandasSelect('REVENUE_3m', fillna_zero=True)),
        # ('ITC_3m', PandasSelect('ITC_3m', fillna_zero=True)),
        # ('VAS_3m', PandasSelect('VAS_3m', fillna_zero=True)),
        ('RENT_CHANNEL_3m', PandasSelect('RENT_CHANNEL_3m', fillna_zero=True)),
        ('ROAM_3m', PandasSelect('ROAM_3m', fillna_zero=True)),
        ('COST_3m', PandasSelect('COST_3m', fillna_zero=True)),
        ('COM_CAT#17_3m', PandasSelect('COM_CAT#17_3m', fillna_zero=True)),
        # ('COM_CAT#18_3m', PandasSelect('COM_CAT#18_3m', fillna_zero=True)),
        ('COM_CAT#19_3m', PandasSelect('COM_CAT#19_3m', fillna_zero=True)),
        ('COM_CAT#20_3m', PandasSelect('COM_CAT#20_3m', fillna_zero=True)),
        ('COM_CAT#21_3m', PandasSelect('COM_CAT#21_3m', fillna_zero=True)),
        # ('COM_CAT#22_3m', PandasSelect('COM_CAT#22_3m', fillna_zero=True)),
        ('COM_CAT#23_3m', PandasSelect('COM_CAT#23_3m', fillna_zero=True)),
        # ('COM_CAT#27_3m', PandasSelect('COM_CAT#27_3m', fillna_zero=True)),
        ('COM_CAT#28_3m', PandasSelect('COM_CAT#28_3m', fillna_zero=True)),
        # ('COM_CAT#29_3m', PandasSelect('COM_CAT#29_3m', fillna_zero=True)),
        ('COM_CAT#30_3m', PandasSelect('COM_CAT#30_3m', fillna_zero=True)),
        ('COM_CAT#31_3m', PandasSelect('COM_CAT#31_3m', fillna_zero=True)),
        # ('COM_CAT#32_3m', PandasSelect('COM_CAT#32_3m', fillna_zero=True)),
        # ('COM_CAT#33_3m', PandasSelect('COM_CAT#33_3m', fillna_zero=True)),

        ('REVENUE_6m', PandasSelect('REVENUE_6m', fillna_zero=True)),
        # ('ITC_6m', PandasSelect('ITC_6m', fillna_zero=True)),
        ('VAS_6m', PandasSelect('VAS_6m', fillna_zero=True)),
        ('RENT_CHANNEL_6m', PandasSelect('RENT_CHANNEL_6m', fillna_zero=True)),
        ('ROAM_6m', PandasSelect('ROAM_6m', fillna_zero=True)),
        ('COST_6m', PandasSelect('COST_6m', fillna_zero=True)),
        ('COM_CAT#17_6m', PandasSelect('COM_CAT#17_6m', fillna_zero=True)),
        ('COM_CAT#18_6m', PandasSelect('COM_CAT#18_6m', fillna_zero=True)),
        ('COM_CAT#19_6m', PandasSelect('COM_CAT#19_6m', fillna_zero=True)),
        ('COM_CAT#20_6m', PandasSelect('COM_CAT#20_6m', fillna_zero=True)),
        ('COM_CAT#21_6m', PandasSelect('COM_CAT#21_6m', fillna_zero=True)),
        # ('COM_CAT#22_6m', PandasSelect('COM_CAT#22_6m', fillna_zero=True)),
        # ('COM_CAT#23_6m', PandasSelect('COM_CAT#23_6m', fillna_zero=True)),
        # ('COM_CAT#27_6m', PandasSelect('COM_CAT#27_6m', fillna_zero=True)),
        # ('COM_CAT#28_6m', PandasSelect('COM_CAT#28_6m', fillna_zero=True)),
        ('COM_CAT#29_6m', PandasSelect('COM_CAT#29_6m', fillna_zero=True)),
        ('COM_CAT#30_6m', PandasSelect('COM_CAT#30_6m', fillna_zero=True)),
        ('COM_CAT#31_6m', PandasSelect('COM_CAT#31_6m', fillna_zero=True)),
        ('COM_CAT#32_6m', PandasSelect('COM_CAT#32_6m', fillna_zero=True)),
        # ('COM_CAT#33_6m', PandasSelect('COM_CAT#33_6m', fillna_zero=True)),

        ('REVENUE_diff_1m', PandasSelect('REVENUE_diff_1m', fillna_zero=True)),
        ('ITC_diff_1m', PandasSelect('ITC_diff_1m', fillna_zero=True)),
        ('VAS_diff_1m', PandasSelect('VAS_diff_1m', fillna_zero=True)),
        ('RENT_CHANNEL_diff_1m', PandasSelect('RENT_CHANNEL_diff_1m', fillna_zero=True)),
        ('ROAM_diff_1m', PandasSelect('ROAM_diff_1m', fillna_zero=True)),
        ('COST_diff_1m', PandasSelect('COST_diff_1m', fillna_zero=True)),
        ('COM_CAT#17_diff_1m', PandasSelect('COM_CAT#17_diff_1m', fillna_zero=True)),
        # ('COM_CAT#18_diff_1m', PandasSelect('COM_CAT#18_diff_1m', fillna_zero=True)),
        ('COM_CAT#19_diff_1m', PandasSelect('COM_CAT#19_diff_1m', fillna_zero=True)),
        ('COM_CAT#20_diff_1m', PandasSelect('COM_CAT#20_diff_1m', fillna_zero=True)),
        ('COM_CAT#21_diff_1m', PandasSelect('COM_CAT#21_diff_1m', fillna_zero=True)),
        ('COM_CAT#22_diff_1m', PandasSelect('COM_CAT#22_diff_1m', fillna_zero=True)),
        ('COM_CAT#23_diff_1m', PandasSelect('COM_CAT#23_diff_1m', fillna_zero=True)),
        ('COM_CAT#27_diff_1m', PandasSelect('COM_CAT#27_diff_1m', fillna_zero=True)),
        # ('COM_CAT#28_diff_1m', PandasSelect('COM_CAT#28_diff_1m', fillna_zero=True)),
        ('COM_CAT#29_diff_1m', PandasSelect('COM_CAT#29_diff_1m', fillna_zero=True)),
        ('COM_CAT#30_diff_1m', PandasSelect('COM_CAT#30_diff_1m', fillna_zero=True)),
        ('COM_CAT#31_diff_1m', PandasSelect('COM_CAT#31_diff_1m', fillna_zero=True)),
        ('COM_CAT#32_diff_1m', PandasSelect('COM_CAT#32_diff_1m', fillna_zero=True)),
        ('COM_CAT#33_diff_1m', PandasSelect('COM_CAT#33_diff_1m', fillna_zero=True)),

        ('REVENUE_diff_2m', PandasSelect('REVENUE_diff_2m', fillna_zero=True)),
        # ('ITC_diff_2m', PandasSelect('ITC_diff_2m', fillna_zero=True)),
        ('VAS_diff_2m', PandasSelect('VAS_diff_2m', fillna_zero=True)),
        ('RENT_CHANNEL_diff_2m', PandasSelect('RENT_CHANNEL_diff_2m', fillna_zero=True)),
        # ('ROAM_diff_2m', PandasSelect('ROAM_diff_2m', fillna_zero=True)),
        ('COST_diff_2m', PandasSelect('COST_diff_2m', fillna_zero=True)),
        ('COM_CAT#17_diff_2m', PandasSelect('COM_CAT#17_diff_2m', fillna_zero=True)),
        ('COM_CAT#18_diff_2m', PandasSelect('COM_CAT#18_diff_2m', fillna_zero=True)),
        # ('COM_CAT#19_diff_2m', PandasSelect('COM_CAT#19_diff_2m', fillna_zero=True)),
        # ('COM_CAT#20_diff_2m', PandasSelect('COM_CAT#20_diff_2m', fillna_zero=True)),
        # ('COM_CAT#21_diff_2m', PandasSelect('COM_CAT#21_diff_2m', fillna_zero=True)),
        ('COM_CAT#22_diff_2m', PandasSelect('COM_CAT#22_diff_2m', fillna_zero=True)),
        ('COM_CAT#23_diff_2m', PandasSelect('COM_CAT#23_diff_2m', fillna_zero=True)),
        # ('COM_CAT#27_diff_2m', PandasSelect('COM_CAT#27_diff_2m', fillna_zero=True)),
        # ('COM_CAT#28_diff_2m', PandasSelect('COM_CAT#28_diff_2m', fillna_zero=True)),
        ('COM_CAT#29_diff_2m', PandasSelect('COM_CAT#29_diff_2m', fillna_zero=True)),
        # ('COM_CAT#30_diff_2m', PandasSelect('COM_CAT#30_diff_2m', fillna_zero=True)),
        # ('COM_CAT#31_diff_2m', PandasSelect('COM_CAT#31_diff_2m', fillna_zero=True)),
        ('COM_CAT#32_diff_2m', PandasSelect('COM_CAT#32_diff_2m', fillna_zero=True)),
        ('COM_CAT#33_diff_2m', PandasSelect('COM_CAT#33_diff_2m', fillna_zero=True)),

        ('REVENUE_diff_3m', PandasSelect('REVENUE_diff_3m', fillna_zero=True)),
        # ('ITC_diff_3m', PandasSelect('ITC_diff_3m', fillna_zero=True)),
        ('VAS_diff_3m', PandasSelect('VAS_diff_3m', fillna_zero=True)),
        ('RENT_CHANNEL_diff_3m', PandasSelect('RENT_CHANNEL_diff_3m', fillna_zero=True)),
        ('ROAM_diff_3m', PandasSelect('ROAM_diff_3m', fillna_zero=True)),
        ('COST_diff_3m', PandasSelect('COST_diff_3m', fillna_zero=True)),
        ('COM_CAT#17_diff_3m', PandasSelect('COM_CAT#17_diff_3m', fillna_zero=True)),
        # ('COM_CAT#18_diff_3m', PandasSelect('COM_CAT#18_diff_3m', fillna_zero=True)),
        ('COM_CAT#19_diff_3m', PandasSelect('COM_CAT#19_diff_3m', fillna_zero=True)),
        # ('COM_CAT#20_diff_3m', PandasSelect('COM_CAT#20_diff_3m', fillna_zero=True)),
        ('COM_CAT#21_diff_3m', PandasSelect('COM_CAT#21_diff_3m', fillna_zero=True)),
        ('COM_CAT#22_diff_3m', PandasSelect('COM_CAT#22_diff_3m', fillna_zero=True)),
        ('COM_CAT#23_diff_3m', PandasSelect('COM_CAT#23_diff_3m', fillna_zero=True)),
        # ('COM_CAT#27_diff_3m', PandasSelect('COM_CAT#27_diff_3m', fillna_zero=True)),
        # ('COM_CAT#28_diff_3m', PandasSelect('COM_CAT#28_diff_3m', fillna_zero=True)),
        # ('COM_CAT#29_diff_3m', PandasSelect('COM_CAT#29_diff_3m', fillna_zero=True)),
        # ('COM_CAT#30_diff_3m', PandasSelect('COM_CAT#30_diff_3m', fillna_zero=True)),
        # ('COM_CAT#31_diff_3m', PandasSelect('COM_CAT#31_diff_3m', fillna_zero=True)),
        ('COM_CAT#32_diff_3m', PandasSelect('COM_CAT#32_diff_3m', fillna_zero=True)),
        ('COM_CAT#33_diff_3m', PandasSelect('COM_CAT#33_diff_3m', fillna_zero=True)),
    ])),
    ('estimator', RandomForestClassifier(n_estimators=100, n_jobs=4))
])


def adversial_train_test_split(train_X, train_y, test_X, topK=500):
    train_X['train'] = 1
    train_X['target'] = train_y
    test_X['train'] = 0
    test_X['target'] = -1

    df = pd.concat((train_X, test_X)).reset_index(drop=True)

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

    train_df = merge_features(train_df, train_feat_df)
    train_df = add_weekday(train_df, 'CONTACT_DATE')
    train_df = add_holidays(train_df, 'CONTACT_DATE')
    train_X = train_df.drop(['CSI', 'CONTACT_DATE', 'SNAP_DATE'], axis=1)
    train_y = train_df['CSI']
    gc.collect()

    test_df = load_csi_test()
    test_feat_df = load_features('test')

    test_df = merge_features(test_df, test_feat_df)
    test_df = add_weekday(test_df, 'CONTACT_DATE')
    test_df = add_holidays(test_df, 'CONTACT_DATE')
    test_X = test_df.drop(['CONTACT_DATE', 'SNAP_DATE'], axis=1)
    gc.collect()

    res_df = adversial_train_test_split(train_X, train_y, test_X, 1000)
    # print(res_df)

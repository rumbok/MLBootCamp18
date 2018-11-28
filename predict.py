from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
import pandas as pd

from data_loading import load_csi_test, load_csi_train, load_features
from data_prepare import merge_features

train_df = load_csi_train()
train_feat_df = load_features('test')

train_df = merge_features(train_df, train_feat_df)
train_y = train_df['CSI']
train_X = train_df.drop(['CSI', 'CONTACT_DATE', 'SNAP_DATE'], axis=1)

clf = LogisticRegressionCV(cv=5, random_state=42, verbose=10, scoring='roc_auc').fit(train_X, train_y)
print(clf.score(train_X, train_y))

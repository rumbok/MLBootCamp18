import pandas as pd


def merge_features(df, features_df):
    features_df = features_df.sort_values('SNAP_DATE', ascending=False).drop_duplicates(['SK_ID'])
    train_df = pd.merge(df, features_df, on='SK_ID', how='left')
    return train_df.fillna(0)
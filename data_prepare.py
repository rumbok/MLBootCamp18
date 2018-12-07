import pandas as pd
from datetime import datetime


def merge_features(df, features_df):
    features_2m_df = features_df[['SK_ID',
                                  'REVENUE', 'ITC', 'VAS', 'RENT_CHANNEL', 'ROAM', 'COST',
                                  'COM_CAT#17', 'COM_CAT#18', 'COM_CAT#19', 'COM_CAT#20', 'COM_CAT#21',
                                  'COM_CAT#22', 'COM_CAT#23', 'COM_CAT#27', 'COM_CAT#28', 'COM_CAT#29',
                                  'COM_CAT#30', 'COM_CAT#31', 'COM_CAT#32', 'COM_CAT#33']]\
        .rolling(2)\
        .mean()\
        .drop_duplicates(['SK_ID'])
    features_3m_df = features_df[['SK_ID',
                                  'REVENUE', 'ITC', 'VAS', 'RENT_CHANNEL', 'ROAM', 'COST',
                                  'COM_CAT#17', 'COM_CAT#18', 'COM_CAT#19', 'COM_CAT#20', 'COM_CAT#21',
                                  'COM_CAT#22', 'COM_CAT#23', 'COM_CAT#27', 'COM_CAT#28', 'COM_CAT#29',
                                  'COM_CAT#30', 'COM_CAT#31', 'COM_CAT#32', 'COM_CAT#33']]\
        .rolling(3)\
        .mean()\
        .drop_duplicates(['SK_ID'])
    features_6m_df = features_df[['SK_ID',
                                  'REVENUE', 'ITC', 'VAS', 'RENT_CHANNEL', 'ROAM', 'COST',
                                  'COM_CAT#17', 'COM_CAT#18', 'COM_CAT#19', 'COM_CAT#20', 'COM_CAT#21',
                                  'COM_CAT#22', 'COM_CAT#23', 'COM_CAT#27', 'COM_CAT#28', 'COM_CAT#29',
                                  'COM_CAT#30', 'COM_CAT#31', 'COM_CAT#32', 'COM_CAT#33']]\
        .rolling(6)\
        .mean()\
        .drop_duplicates(['SK_ID'])

    features_diff_1m_df = features_df[['SK_ID',
                                       'REVENUE', 'ITC', 'VAS', 'RENT_CHANNEL', 'ROAM', 'COST',
                                       'COM_CAT#17', 'COM_CAT#18', 'COM_CAT#19', 'COM_CAT#20', 'COM_CAT#21',
                                       'COM_CAT#22', 'COM_CAT#23', 'COM_CAT#27', 'COM_CAT#28', 'COM_CAT#29',
                                       'COM_CAT#30', 'COM_CAT#31', 'COM_CAT#32', 'COM_CAT#33']] \
        .set_index('SK_ID') \
        .diff(periods=-1) \
        .reset_index(drop=False)
    features_diff_2m_df = features_diff_1m_df \
        .shift(-1) \
        .drop_duplicates(['SK_ID'])
    features_diff_3m_df = features_diff_1m_df \
        .shift(-2) \
        .drop_duplicates(['SK_ID'])
    features_diff_1m_df = features_diff_1m_df\
        .drop_duplicates(['SK_ID'])

    features_df = features_df\
        .drop_duplicates(['SK_ID'])

    train_df = pd.merge(df, features_df, on='SK_ID', how='left')

    train_df = pd.merge(train_df, features_2m_df, on='SK_ID', how='left', suffixes=('', '_2m'))
    train_df = pd.merge(train_df, features_3m_df, on='SK_ID', how='left', suffixes=('', '_3m'))
    train_df = pd.merge(train_df, features_6m_df, on='SK_ID', how='left', suffixes=('', '_6m'))

    train_df = pd.merge(train_df, features_diff_1m_df, on='SK_ID', how='left', suffixes=('', '_diff_1m'))
    train_df = pd.merge(train_df, features_diff_2m_df, on='SK_ID', how='left', suffixes=('', '_diff_2m'))
    train_df = pd.merge(train_df, features_diff_3m_df, on='SK_ID', how='left', suffixes=('', '_diff_3m'))
    return train_df.fillna(0)


def add_weekday(df, data_field: str):
    df[data_field + '_WEEKDAY'] = df[data_field].dt.weekday
    return df


HOLIDAYS = [
    datetime(2018, 2, 23),
    datetime(2018, 3, 8),
    datetime(2018, 3, 9),
    datetime(2018, 3, 15),
    # datetime(2018, 4, 30),
    # datetime(2018, 5, 1),
    # datetime(2018, 5, 2),
    # datetime(2018, 5, 9),
    # datetime(2018, 5, 15),
]


def add_holidays(df, data_field: str):
    for i, date in enumerate(HOLIDAYS):
        df[data_field + f'_{i}_HOLIDAYS'] = ((df[data_field] - HOLIDAYS[i]).dt.days).clip(-1, 3)
    return df


features = [
    'COM_CAT#1',
        # 'COM_CAT#2',
        'COM_CAT#3',
    'BASE_TYPE',
        # 'ACT',
        # 'ARPU_GROUP',
    'COM_CAT#7',
        # 'COM_CAT#8',
        'DEVICE_TYPE_ID',
        # 'INTERNET_TYPE_ID',
        'REVENUE',
        'ITC',
    'VAS',
    'RENT_CHANNEL',
    'ROAM',
    'COST',
    'COM_CAT#17',
        'COM_CAT#18',
    'COM_CAT#19',
    'COM_CAT#20',
        'COM_CAT#21',
    'COM_CAT#22',
        # 'COM_CAT#23',
        'COM_CAT#25',
        'COM_CAT#26',
        'COM_CAT#27',
        'COM_CAT#28',
    'COM_CAT#29',
        # 'COM_CAT#30',
    'COM_CAT#31',
    # 'COM_CAT#32',
    'COM_CAT#33',
        # 'COM_CAT#34',

        # 'CONTACT_DATE_WEEKDAY',

    # 'CONTACT_DATE_0_HOLIDAYS',
        # 'CONTACT_DATE_1_HOLIDAYS',
        # 'CONTACT_DATE_2_HOLIDAYS',
    # 'CONTACT_DATE_3_HOLIDAYS',

    'REVENUE_2m',
    'ITC_2m',
    # 'VAS_2m',
    'RENT_CHANNEL_2m',
    'ROAM_2m',
    'COST_2m',
    # 'COM_CAT#17_2m',
    'COM_CAT#18_2m',
    'COM_CAT#19_2m',
    'COM_CAT#20_2m',
    'COM_CAT#21_2m',
    'COM_CAT#22_2m',
    'COM_CAT#23_2m',
    'COM_CAT#27_2m',
    'COM_CAT#28_2m',
    'COM_CAT#29_2m',
    'COM_CAT#30_2m',
    'COM_CAT#31_2m',
    # 'COM_CAT#32_2m',
    'COM_CAT#33_2m',

    'REVENUE_3m',
    # 'ITC_3m',
    # 'VAS_3m',
    'RENT_CHANNEL_3m',
    'ROAM_3m',
    'COST_3m',
    'COM_CAT#17_3m',
    # 'COM_CAT#18_3m',
    'COM_CAT#19_3m',
    'COM_CAT#20_3m',
    'COM_CAT#21_3m',
    # 'COM_CAT#22_3m',
    'COM_CAT#23_3m',
    # 'COM_CAT#27_3m',
    'COM_CAT#28_3m',
    # 'COM_CAT#29_3m',
    'COM_CAT#30_3m',
    'COM_CAT#31_3m',
    # 'COM_CAT#32_3m',
    # 'COM_CAT#33_3m',

    'REVENUE_6m',
    # 'ITC_6m',
    'VAS_6m',
    'RENT_CHANNEL_6m',
    'ROAM_6m',
    'COST_6m',
    'COM_CAT#17_6m',
    'COM_CAT#18_6m',
    'COM_CAT#19_6m',
    'COM_CAT#20_6m',
    'COM_CAT#21_6m',
    # 'COM_CAT#22_6m',
    # 'COM_CAT#23_6m',
    # 'COM_CAT#27_6m',
    # 'COM_CAT#28_6m',
    'COM_CAT#29_6m',
    'COM_CAT#30_6m',
    'COM_CAT#31_6m',
    'COM_CAT#32_6m',
    # 'COM_CAT#33_6m',

    'REVENUE_diff_1m',
    'ITC_diff_1m',
    'VAS_diff_1m',
    'RENT_CHANNEL_diff_1m',
    'ROAM_diff_1m',
    'COST_diff_1m',
    'COM_CAT#17_diff_1m',
    # 'COM_CAT#18_diff_1m',
    'COM_CAT#19_diff_1m',
    'COM_CAT#20_diff_1m',
    'COM_CAT#21_diff_1m',
    'COM_CAT#22_diff_1m',
    'COM_CAT#23_diff_1m',
    'COM_CAT#27_diff_1m',
    # 'COM_CAT#28_diff_1m',
    'COM_CAT#29_diff_1m',
    'COM_CAT#30_diff_1m',
    'COM_CAT#31_diff_1m',
    'COM_CAT#32_diff_1m',
    'COM_CAT#33_diff_1m',

    'REVENUE_diff_2m',
    # 'ITC_diff_2m',
    'VAS_diff_2m',
    'RENT_CHANNEL_diff_2m',
    # 'ROAM_diff_2m',
    'COST_diff_2m',
    'COM_CAT#17_diff_2m',
    'COM_CAT#18_diff_2m',
    # 'COM_CAT#19_diff_2m',
    # 'COM_CAT#20_diff_2m',
    # 'COM_CAT#21_diff_2m',
    'COM_CAT#22_diff_2m',
    'COM_CAT#23_diff_2m',
    # 'COM_CAT#27_diff_2m',
    # 'COM_CAT#28_diff_2m',
    'COM_CAT#29_diff_2m',
    # 'COM_CAT#30_diff_2m',
    # 'COM_CAT#31_diff_2m',
    'COM_CAT#32_diff_2m',
    'COM_CAT#33_diff_2m',

    'REVENUE_diff_3m',
    # 'ITC_diff_3m',
    'VAS_diff_3m',
    'RENT_CHANNEL_diff_3m',
    'ROAM_diff_3m',
    'COST_diff_3m',
    'COM_CAT#17_diff_3m',
    # 'COM_CAT#18_diff_3m',
    'COM_CAT#19_diff_3m',
    # 'COM_CAT#20_diff_3m',
    'COM_CAT#21_diff_3m',
    'COM_CAT#22_diff_3m',
    'COM_CAT#23_diff_3m',
    # 'COM_CAT#27_diff_3m',
    # 'COM_CAT#28_diff_3m',
    # 'COM_CAT#29_diff_3m',
    # 'COM_CAT#30_diff_3m',
    # 'COM_CAT#31_diff_3m',
    'COM_CAT#32_diff_3m',
    'COM_CAT#33_diff_3m',
]
categorical = [
    'COM_CAT#1',
        'COM_CAT#2',
        'COM_CAT#3',
    'BASE_TYPE',
        'ACT',
        'ARPU_GROUP',
    'COM_CAT#7',
        # 'COM_CAT#8',
        'DEVICE_TYPE_ID',
        'INTERNET_TYPE_ID',
        'COM_CAT#25',
        'COM_CAT#26',
        # 'COM_CAT#34',

        # 'CONTACT_DATE_WEEKDAY',

    'CONTACT_DATE_0_HOLIDAYS',
        # 'CONTACT_DATE_1_HOLIDAYS',
        'CONTACT_DATE_2_HOLIDAYS',
    'CONTACT_DATE_3_HOLIDAYS',
]
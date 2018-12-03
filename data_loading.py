import pandas as pd
import numpy as np
import os

TRAIN_DIR = './dataset/train'
TEST_DIR = './dataset/test'
BS_DIR = './dataset/'
CACHE_DIR = './cache/'


def load_consumption(tp: str):
    if tp == 'test':
        filepath = os.path.join(TEST_DIR, 'subs_bs_consumption_{}.csv'.format(tp))
    else:
        filepath = os.path.join(TRAIN_DIR, 'subs_bs_consumption_{}.csv'.format(tp))
    df = pd.read_csv(filepath,
                     delimiter=';',
                     decimal=',',
                     dtype={
                         'SK_ID': np.uint16,
                         'CELL_LAC_ID': np.uint32,
                         'MON': str,
                         'SUM_MINUTES': np.float16,
                         'SUM_DATA_MB': np.float16,
                         'SUM_DATA_MIN': np.float16,
                     })
    df['MON'] = pd.to_datetime(df['MON'] + '.2002',
                               dayfirst=True,
                               format='%d.%m.%Y',
                               infer_datetime_format=True,
                               cache=True)
    return df


def load_data_session(tp: str):
    if tp == 'test':
        filepath = os.path.join(TEST_DIR, 'subs_bs_data_session_{}.csv'.format(tp))
    else:
        filepath = os.path.join(TRAIN_DIR, 'subs_bs_data_session_{}.csv'.format(tp))
    df = pd.read_csv(filepath,
                     delimiter=';',
                     decimal=',',
                     dtype={
                         'SK_ID': np.uint16,
                         'CELL_LAC_ID': np.uint32,
                         'DATA_VOL_MB': np.float16,
                         'START_TIME': str,
                     })
    df['START_TIME'] = pd.to_datetime(df['START_TIME'] + ' 2002',
                                      dayfirst=True,
                                      format='%d.%m %H:%M:%S %Y',
                                      infer_datetime_format=True,
                                      cache=True)
    return df


def load_voice_session(tp: str):
    if tp == 'test':
        filepath = os.path.join(TEST_DIR, 'subs_bs_voice_session_{}.csv'.format(tp))
    else:
        filepath = os.path.join(TRAIN_DIR, 'subs_bs_voice_session_{}.csv'.format(tp))
    df = pd.read_csv(filepath,
                     delimiter=';',
                     decimal=',',
                     dtype={
                         'SK_ID': np.uint16,
                         'CELL_LAC_ID': np.uint32,
                         'VOICE_DUR_MIN': np.float16,
                         'START_TIME': str,
                     })
    df['START_TIME'] = pd.to_datetime(df['START_TIME'] + ' 2002',
                                      dayfirst=True,
                                      format='%d.%m %H:%M:%S %Y',
                                      infer_datetime_format=True,
                                      cache=True)
    return df


def load_csi_train():
    df = pd.read_csv(os.path.join(TRAIN_DIR, 'subs_csi_train.csv'),
                     delimiter=';',
                     dtype={
                         'SK_ID': np.uint16,
                         'CSI': np.uint8,
                         'CONTACT_DATE': str
                     })
    df['CONTACT_DATE'] = pd.to_datetime(df['CONTACT_DATE'] + '.2002',
                                        dayfirst=True,
                                        format='%d.%m',
                                        infer_datetime_format=True,
                                        cache=True)
    return df


def load_csi_test():
    df = pd.read_csv(os.path.join(TEST_DIR, 'subs_csi_test.csv'),
                     delimiter=';',
                     dtype={
                         'SK_ID': np.uint16,
                         'CONTACT_DATE': str
                     })
    df['CONTACT_DATE'] = pd.to_datetime(df['CONTACT_DATE'] + '.2002',
                                        dayfirst=True,
                                        format='%d.%m.%Y',
                                        infer_datetime_format=True,
                                        cache=True)
    return df


def load_features(tp: str):
    if tp == 'test':
        filepath = os.path.join(TEST_DIR, 'subs_features_{}.csv'.format(tp))
    else:
        filepath = os.path.join(TRAIN_DIR, 'subs_features_{}.csv'.format(tp))
    df = pd.read_csv(filepath,
                     delimiter=';',
                     decimal=',',
                     dtype={
                         'SNAP_DATE': str,
                         'COM_CAT#1': np.uint8,
                         'SK_ID': np.uint16,
                         'COM_CAT#2': np.uint8,
                         'COM_CAT#3': np.uint8,
                         'BASE_TYPE': np.uint8,
                         'ACT': np.uint8,
                         'ARPU_GROUP': np.float16,
                         'COM_CAT#7': np.uint8,
                         'COM_CAT#8': np.float32,
                         'DEVICE_TYPE_ID': np.float16,
                         'INTERNET_TYPE_ID': np.float16,
                         'REVENUE': np.float16,
                         'ITC': np.float16,
                         'VAS': np.float16,
                         'RENT_CHANNEL': np.float16,
                         'ROAM': np.float16,
                         'COST': np.float16,
                         'COM_CAT#17': np.float16,
                         'COM_CAT#18': np.float16,
                         'COM_CAT#19': np.float16,
                         'COM_CAT#20': np.float16,
                         'COM_CAT#21': np.float16,
                         'COM_CAT#22': np.float16,
                         'COM_CAT#23': np.float16,
                         'COM_CAT#24': str,
                         'COM_CAT#25': np.uint8,
                         'COM_CAT#26': np.uint8,
                         'COM_CAT#27': np.float16,
                         'COM_CAT#28': np.float16,
                         'COM_CAT#29': np.float16,
                         'COM_CAT#30': np.float16,
                         'COM_CAT#31': np.float16,
                         'COM_CAT#32': np.float16,
                         'COM_CAT#33': np.float16,
                         'COM_CAT#34': np.float16,
                     })
    df['SNAP_DATE'] = pd.to_datetime(df['SNAP_DATE'],
                                     dayfirst=True,
                                     format='%d.%m.%y',
                                     infer_datetime_format=True,
                                     cache=True)
    df['COM_CAT#24'] = pd.to_datetime(df['COM_CAT#24'] + '.2001',
                                      dayfirst=True,
                                      format='%d.%m.%y',
                                      infer_datetime_format=True,
                                      cache=True)
    df['ARPU_GROUP'] = df['ARPU_GROUP'].fillna(0).astype(np.uint8)
    df['COM_CAT#8'] = df['COM_CAT#8'].fillna(0).astype(np.uint16)
    df['DEVICE_TYPE_ID'] = df['DEVICE_TYPE_ID'].fillna(0).astype(np.uint8)
    df['INTERNET_TYPE_ID'] = df['INTERNET_TYPE_ID'].fillna(0).astype(np.uint8)
    df['COM_CAT#34'] = df['COM_CAT#34'].fillna(0).astype(np.uint8)
    return df


if __name__ == '__main__':
    train_df = load_features('train')
    test_df = load_features('test')
    print(test_df.info())

    field = 'RENT_CHANNEL'
    train_vc = train_df[field].value_counts()
    test_vc = test_df[field].value_counts()
    print(train_vc)
    print(test_vc)
    print('Not in test', set(train_vc.index) - set(test_vc.index))
    print('Not in train', set(test_vc.index) - set(train_vc.index))

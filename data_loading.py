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
    df['CONTACT_DATE'] = pd.to_datetime(df['CONTACT_DATE'],
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

    return df


# def load_train():
#     print('loading all train')
#     if os.path.exists(os.path.join(CACHE_DIR, 'train_all.feather')):
#         train = pd.read_feather(os.path.join(CACHE_DIR, 'train_all.feather'))
#     else:
#         train = pd.read_csv(os.path.join(INPUT_DIR, 'train.csv.zip'),
#                             parse_dates=['attributed_time', 'click_time'],
#                             dtype=dtypes)
#         train['is_attributed'] = train['is_attributed'].fillna(0).astype('uint8')
#         train['att_delta'] = (train['attributed_time'] - train['click_time']) \
#             .dt \
#             .total_seconds() \
#             .fillna(0.0) \
#             .astype('uint32')
#         del train['attributed_time']
#         train.dropna(subset=['click_time'], axis=0, how='any', inplace=True)
#         dt = train['click_time'].dt
#         train['day'] = dt.day.astype('uint8')
#         train['hour'] = dt.hour.astype('uint8')
#         del dt
#         print(train.info(verbose=True, memory_usage=True, null_counts=True))
#         train.to_feather(os.path.join(CACHE_DIR, 'train_all.feather'))
#     return train


if __name__ == '__main__':
    df = load_features('train')
    print(df.info())

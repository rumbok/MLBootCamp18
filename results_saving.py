from time import strftime, gmtime
import pandas as pd


def print_gain(model_lgb):
    print("features importance...")
    gain = model_lgb.feature_importance('gain')
    ft = pd.DataFrame({'feature': model_lgb.feature_name(),
                       'split': model_lgb.feature_importance('split'),
                       'gain': 100 * gain / gain.sum()}).sort_values('gain', ascending=False)
    print(ft.head(50))
    ft.to_csv('importance_lightgbm.csv', index=True)


def save_results(train_days, valid_days, len_train, len_valid, features, categories, lgb_params, n_estimators,
                 train_auc, valid_auc):
    with open('val_experiments.txt', 'a+') as f:
        f.write(strftime("%Y-%m-%d %H:%M:%S\n", gmtime()))
        f.write('train days: {}, size={}\n'.format(train_days, len_train))
        f.write('valid day: {}, size={}\n'.format(valid_days, len_valid))
        f.write('features: {}\n'.format(features))
        f.write('categorical: {}\n'.format(categories))
        f.write('Light GBM params: {}\n'.format(lgb_params))
        f.write('Model Report\n')
        f.write('n_estimators: {}\n'.format(n_estimators))
        f.write("AUC: train={} valid={}".format(train_auc, valid_auc))
        f.write('\n\n')
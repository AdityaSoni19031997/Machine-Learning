import numpy as np
import pandas as pd


def label_uniques(cols, label):
    for col in cols:
        train_unique_idx = X_train[~X_train[col].isin(X_test[col])].index
        test_unique_idx = X_test[~X_test[col].isin(X_train[col])].index
        
        X_train.loc[train_unique_idx, col] = label       
        print('X_train: {} values in {} are labeled.'.format(len(train_unique_idx), col))
        X_test.loc[test_unique_idx, col] = label     
        print('X_test {} values in {} are labeled.'.format(len(test_unique_idx), col))
        
def scale_by_time(periods, cols):
    for period in periods:
        for df in [X_train, X_test]:
            for col in cols:
                temp_min = df.groupby([period])[col].agg(['min']).reset_index()
                temp_min.index = temp_min[period].values
                temp_min = df[period].map(temp_min['min'].to_dict())

                temp_max = df.groupby([period])[col].agg(['max']).reset_index()
                temp_max.index = temp_max[period].values
                temp_max = df[period].map(temp_max['max'].to_dict())

                new_dcol_name = '{}_{}_MinMaxScaled'.format(col, period)                
                df[new_dcol_name] = (df[col] - temp_min) / (temp_max - temp_min)
                
                not_zero = df[new_dcol_name] > 0
                df.loc[df[not_zero].index, new_dcol_name] = np.log10(df[not_zero][new_dcol_name])

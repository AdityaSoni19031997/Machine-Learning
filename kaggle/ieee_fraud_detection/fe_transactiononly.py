from datetime import datetime, timedelta
import numpy as np
import pandas as pd


CAT_FCOLS = ['card2', 'card3', 'card5', 'addr1', 'addr2', 'dist1', 'dist2']
C_FCOLS = [f'C{i}' for i in range(1, 15)]
D_FCOLS = [f'D{i}' for i in range(1, 16)]
V_FCOLS = [f'V{i}' for i in range(1, 340)]
FLOAT64_TCOLS = CAT_FCOLS + C_FCOLS + D_FCOLS + V_FCOLS

X_train = pd.read_csv('../input/ieee-fraud-detection/train_transaction.csv', dtype=dict.fromkeys(FLOAT64_TCOLS, np.float32))
X_test = pd.read_csv('../input/ieee-fraud-detection/test_transaction.csv', dtype=dict.fromkeys(FLOAT64_TCOLS, np.float32))
print('Number of Training Examples = {}'.format(X_train.shape[0]))
print('Number of Test Examples = {}\n'.format(X_test.shape[0]))
print('Training X Shape with Only Transaction = {}'.format(X_train.shape))
print('Test X Shape with Only Transaction = {}\n'.format(X_test.shape))
print('Training Set Memory Usage = {:.2f} MB'.format(X_train.memory_usage().sum() / 1024**2))
print('Test Set Memory Usage = {:.2f} MB\n'.format(X_test.memory_usage().sum() / 1024**2))


START_DATE = datetime.strptime('2017-12-01', '%Y-%m-%d')
# START_DATE = datetime.strptime('2017-11-01', '%Y-%m-%d')

for df in [X_train, X_test]:
    # TransactionDT converted to a timestamp
    df['TransactionDate'] = (df['TransactionDT'] - 86400).apply(lambda x: (START_DATE + timedelta(seconds=x)))

    # Time features for aggregation and grouping
    df['Minute'] = df['TransactionDate'].dt.minute.values
    df['Hour'] = df['TransactionDate'].dt.hour.values
    df['Day'] = df['TransactionDate'].dt.day.values
    df['DayOfWeek'] = df['TransactionDate'].dt.dayofweek.values
    df['DayOfYear'] = df['TransactionDate'].dt.dayofyear.values
    df['Week'] = df['TransactionDate'].dt.week.values
    df['Month'] = df['TransactionDate'].dt.month.values

    # D9 is Hour divided by 24, so this will fill the NaNs of D9
    df['D9'] = df['Hour'] / 24

    # TransactionAmtCents
    df['TransactionAmtCents'] = df['TransactionAmt'] - df['TransactionAmt'].astype('int')


# SIMPLE AGGREGATIONS AND COMBINATIONS
D_COLS = [f'D{i}' for i in range(1, 16) if i != 9]

for df in [X_train, X_test]:

    # Feature Combination
    df['Card_ID1'] = df['card1'].astype(str) + '_' + df['card2'].astype(str)
    df['Card_ID2'] = df['Card_ID1'].astype(str) + '_' + df['card3'].astype(str) + '_' + df['card5'].astype(str)
    df['Card_ID3'] = df['Card_ID2'].astype(str) + '_' + df['addr1'].astype(str) + '_' + df['addr2'].astype(str)
    df['Item_ID'] = df['TransactionAmt'].astype(str) + '_' + df['ProductCD'].astype(str)
    df['PAccount_ID'] = df['addr1'].astype(str) + '_' + df['addr2'].astype(str) + '_' + df['P_emaildomain'].astype(str)
    df['RAccount_ID'] = df['addr1'].astype(str) + '_' + df['addr2'].astype(str) + '_' + df['R_emaildomain'].astype(str)
    df['PR_emaildomain'] = df['P_emaildomain'].astype(str) + '_' + df['R_emaildomain'].astype(str)

    # D unique count
    df['D_Uniques'] = df[D_COLS].nunique(axis=1)

    # D normalized
    for d in D_COLS:
        df[f'{d}_Week_Norm'] = df[d] - df['Week'].map(
            pd.concat([X_train[[d, 'Week']], X_test[[d, 'Week']]], ignore_index=True).groupby('Week')[d].mean())

    # V-Block Aggregation
    for block in [(1, 12), (12, 35), (35, 53), (53, 75), (75, 95), (95, 138), (138, 167), (167, 217), (217, 279),
                  (279, 322), (322, 340)]:
        df['V{}-V{}_Sum'.format(*block)] = df[['V{}'.format(i) for i in range(*block)]].sum(axis=1)
        df['V{}-V{}_Mean'.format(*block)] = df[['V{}'.format(i) for i in range(*block)]].mean(axis=1)
        df['V{}-V{}_Std'.format(*block)] = df[['V{}'.format(i) for i in range(*block)]].std(axis=1)


# CONTINUOUS/CATEGORICAL GROUPING AGGREGATIONS
C_COLS = [f'C{i}' for i in range(1, 15)]
DN_COLS = [f'D{i}_Week_Norm' for i in range(1, 16) if i != 9]
CONT_COLS = ['TransactionAmt', 'dist1', 'dist2'] + C_COLS + DN_COLS
CAT_COLS = ['card1', 'card2', 'card3', 'card5', 'Card_ID1', 'Card_ID2', 'Card_ID3', 'addr1', 'P_emaildomain',
            'R_emaildomain', 'PAccount_ID', 'RAccount_ID', 'PR_emaildomain']
AGG_TYPES = ['std', 'mean', 'sum']
# CREATES len(CONT_COLS) * len(CAT_COLS) * len(AGG_TYPES) FEATURES
# CHANGE FEATURES HERE IN CASE OF MEMORY ISSUES

for cat_col in CAT_COLS:
    for cont_col in CONT_COLS:
        for agg_type in AGG_TYPES:
            new_col_name = cat_col + f'_{cont_col}_' + agg_type.capitalize()

            temp_df = pd.concat([X_train[[cat_col, cont_col]], X_test[[cat_col, cont_col]]])
            temp_df = temp_df.groupby([cat_col])[cont_col].agg([agg_type]).reset_index().rename(columns={agg_type: new_col_name})
            temp_df.index = list(temp_df[cat_col])
            temp_df = temp_df[new_col_name].to_dict()

            X_train[new_col_name] = X_train[cat_col].map(temp_df).astype(np.float32)
            X_test[new_col_name] = X_test[cat_col].map(temp_df).astype(np.float32)


# FREQUENCY ENCODING
for df in [X_train, X_test]:

    # Frequency Encoding
    for col in CAT_COLS + ['TransactionAmt', 'TransactionAmtCents'] + C_COLS:
        df[f'{col}_VC'] = df[col].map(
            pd.concat([X_train[col], X_test[col]], ignore_index=True).value_counts(dropna=False))


# DOWNCASTING INT COLS
INT64_COLS = [col for col in X_test.columns if X_test[col].dtype == 'int64']

for df in [X_train, X_test]:
    for col in INT64_COLS:
        c_min = df[col].min()
        c_max = df[col].max()

        if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
            df[col] = df[col].astype(np.int8)
        elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
            df[col] = df[col].astype(np.int16)
        elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
            df[col] = df[col].astype(np.int32)
        elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
            df[col] = df[col].astype(np.int64)

print('{} columns are downcasted.'.format(len(INT64_COLS)))

X_train.to_csv('train.csv', index=False)
X_test.to_csv('test.csv', index=False)
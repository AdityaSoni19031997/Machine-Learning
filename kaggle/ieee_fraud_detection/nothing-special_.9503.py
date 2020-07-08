
# coding: utf-8

# In[1]:


import gc
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 300)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

import lightgbm as lgb

SEED = 42

float_cols = [
 'card2', 'card3', 'card5', 'addr1', 'addr2', 'dist1', 'dist2', 'C1',
 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'D1',
 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15',
 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'V1',
 'V2','V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16',
 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'V29', 'V30',
 'V31', 'V32', 'V33', 'V34', 'V35', 'V36', 'V37', 'V38', 'V39', 'V40', 'V41', 'V42', 'V43', 'V44', 'V45', 'V46', 'V47', 'V48', 'V49',
 'V50', 'V51', 'V52', 'V53', 'V54', 'V55', 'V56', 'V57', 'V58', 'V59', 'V60', 'V61', 'V62', 'V63', 'V64', 'V65', 'V66', 'V67',
 'V68', 'V69', 'V70', 'V71', 'V72', 'V73', 'V74','V75', 'V76', 'V77', 'V78', 'V79', 'V80', 'V81', 'V82', 'V83', 'V84', 'V85',
 'V86', 'V87', 'V88', 'V89', 'V90', 'V91', 'V92', 'V93', 'V94', 'V95', 'V96', 'V97', 'V98', 'V99', 'V100', 'V101', 'V102', 'V103',
 'V104', 'V105', 'V106', 'V107', 'V108', 'V109', 'V110', 'V111', 'V112', 'V113', 'V114', 'V115', 'V116', 'V117', 'V118', 'V119',
 'V120', 'V121', 'V122', 'V123', 'V124', 'V125', 'V126', 'V127', 'V128', 'V129', 'V130', 'V131', 'V132', 'V133', 'V134', 'V135',
 'V136', 'V137', 'V138', 'V139', 'V140', 'V141', 'V142', 'V143', 'V144', 'V145', 'V146', 'V147', 'V148', 'V149', 'V150', 'V151',
 'V152', 'V153', 'V154', 'V155', 'V156', 'V157', 'V158', 'V159', 'V160', 'V161', 'V162', 'V163', 'V164', 'V165', 'V166', 'V167',
 'V168', 'V169', 'V170', 'V171', 'V172', 'V173', 'V174', 'V175', 'V176', 'V177', 'V178', 'V179', 'V180', 'V181', 'V182', 'V183',
 'V184', 'V185', 'V186', 'V187', 'V188', 'V189', 'V190', 'V191', 'V192', 'V193', 'V194', 'V195', 'V196', 'V197', 'V198', 'V199',
 'V200', 'V201', 'V202', 'V203', 'V204', 'V205', 'V206', 'V207', 'V208', 'V209', 'V210', 'V211', 'V212', 'V213', 'V214', 'V215',
 'V216', 'V217', 'V218', 'V219', 'V220', 'V221', 'V222', 'V223', 'V224', 'V225', 'V226', 'V227', 'V228', 'V229', 'V230', 'V231',
 'V232', 'V233', 'V234', 'V235', 'V236', 'V237', 'V238', 'V239', 'V240', 'V241', 'V242', 'V243', 'V244', 'V245', 'V246', 'V247',
 'V248', 'V249', 'V250', 'V251', 'V252', 'V253', 'V254', 'V255', 'V256', 'V257', 'V258', 'V259', 'V260', 'V261', 'V262', 'V263',
 'V264', 'V265', 'V266', 'V267', 'V268', 'V269', 'V270', 'V271', 'V272', 'V273', 'V274', 'V275', 'V276', 'V277', 'V278', 'V279',
 'V280', 'V281', 'V282', 'V283', 'V284', 'V285', 'V286', 'V287', 'V288', 'V289', 'V290', 'V291', 'V292', 'V293', 'V294', 'V295',
 'V296', 'V297', 'V298', 'V299', 'V300', 'V301', 'V302', 'V303', 'V304', 'V305', 'V306', 'V307', 'V308', 'V309', 'V310', 'V311',
 'V312', 'V313', 'V314', 'V315', 'V316', 'V317', 'V318', 'V319', 'V320', 'V321', 'V322', 'V323', 'V324', 'V325', 'V326', 'V327',
 'V328', 'V329', 'V330', 'V331', 'V332', 'V333', 'V334', 'V335', 'V336', 'V337', 'V338', 'V339', 
 'id_01', 'id_02', 'id_03', 'id_04', 'id_05', 'id_06', 'id_07', 'id_08', 'id_09', 'id_10', 'id_11', 'id_12', 'id_13', 'id_14',
 'id_15', 'id_16', 'id_17', 'id_18', 'id_19', 'id_20', 'id_21', 'id_22', 'id_23', 'id_24', 'id_25', 'id_26', 'id_27', 'id_28', 'id_29', 'id_32',
 'id_34', 'id_35', 'id_36', 'id_37', 'id_38',
 'V1-V12_Sum', 'V1-V12_Mean', 'V1-V12_Std',
 'V12-V35_Sum', 'V12-V35_Mean', 'V12-V35_Std',
 'V35-V53_Sum', 'V35-V53_Mean', 'V35-V53_Std',
  'V53-V75_Sum', 'V53-V75_Mean', 'V53-V75_Std',
 'V75-V95_Sum', 'V75-V95_Mean', 'V75-V95_Std',
 'V95-V138_Sum', 'V95-V138_Mean', 'V95-V138_Std',
 'V138-V167_Sum', 'V138-V167_Mean', 'V138-V167_Std',
 'V167-V217_Sum', 'V167-V217_Mean', 'V167-V217_Std',
 'V217-V279_Sum', 'V217-V279_Mean', 'V217-V279_Std',
 'V279-V322_Sum', 'V279-V322_Mean', 'V279-V322_Std',
 'V322-V340_Sum', 'V322-V340_Mean', 'V322-V340_Std',
 'card1_TransactionAmt_Mean', 'card1_TransactionAmt_Std', 'card1_TransactionAmt_Sum', 'card2_TransactionAmt_Mean',
 'card2_TransactionAmt_Std', 'card2_TransactionAmt_Sum',
 'card3_TransactionAmt_Mean', 'card3_TransactionAmt_Std', 'card3_TransactionAmt_Sum',
 'card4_TransactionAmt_Mean', 'card4_TransactionAmt_Std', 'card4_TransactionAmt_Sum',
 'card5_TransactionAmt_Mean', 'card5_TransactionAmt_Std', 'card5_TransactionAmt_Sum',
 'card6_TransactionAmt_Mean', 'card6_TransactionAmt_Std', 'card6_TransactionAmt_Sum',
 'Card_ID1_TransactionAmt_Mean', 'Card_ID1_TransactionAmt_Std',
 'Card_ID1_TransactionAmt_Sum', 'Card_ID2_TransactionAmt_Mean',
 'Card_ID2_TransactionAmt_Std', 'Card_ID2_TransactionAmt_Sum',
 'Card_ID3_TransactionAmt_Mean' 'Card_ID3_TransactionAmt_Std', 'Card_ID3_TransactionAmt_Sum',
 'addr1_TransactionAmt_Mean', 'addr1_TransactionAmt_Std', 'addr1_TransactionAmt_Sum',
 'addr2_TransactionAmt_Mean', 'addr2_TransactionAmt_Std', 'addr2_TransactionAmt_Sum',
 'P_emaildomain_TransactionAmt_Mean', 'P_emaildomain_TransactionAmt_Std', 'P_emaildomain_TransactionAmt_Sum',
 'R_emaildomain_TransactionAmt_Mean', 'R_emaildomain_TransactionAmt_Std', 'R_emaildomain_TransactionAmt_Sum',
 'PAccount_ID_TransactionAmt_Mean', 'PAccount_ID_TransactionAmt_Std', 'PAccount_ID_TransactionAmt_Sum',
 'RAccount_ID_TransactionAmt_Mean', 'RAccount_ID_TransactionAmt_Std', 'RAccount_ID_TransactionAmt_Sum',
 'PR_emaildomain_TransactionAmt_Mean', 'PR_emaildomain_TransactionAmt_Std', 'PR_emaildomain_TransactionAmt_Sum',
 'DeviceInfo_TransactionAmt_Mean', 'DeviceInfo_TransactionAmt_Std', 'DeviceInfo_TransactionAmt_Sum',
 'id_30_TransactionAmt_Mean', 'id_30_TransactionAmt_Std', 'id_30_TransactionAmt_Sum',
 'id_31_TransactionAmt_Mean', 'id_31_TransactionAmt_Std', 'id_31_TransactionAmt_Sum',
 'id_20_TransactionAmt_Mean', 'id_20_TransactionAmt_Std', 'id_20_TransactionAmt_Sum',
 'Device_ID_TransactionAmt_Mean', 'Device_ID_TransactionAmt_Std', 'Device_ID_TransactionAmt_Sum',
 'ParsingError', 'BrowserUpToDate'
 ]

X_train = pd.read_csv('__preprocessed_train.csv', dtype=dict.fromkeys(float_cols, np.float32))
X_test = pd.read_csv('__preprocessed_test.csv', dtype=dict.fromkeys(float_cols, np.float32))
y_train = X_train['isFraud'].copy()
X_train.drop(columns=['isFraud'], inplace=True)

print('Number of Training Examples = {}'.format(X_train.shape[0]))
print('Number of Test Examples = {}'.format(X_test.shape[0]))
print('Training Set Memory Usage = {:.2f} MB'.format(X_train.memory_usage().sum() / 1024**2))
print('Test Set Memory Usage = {:.2f} MB\n'.format(X_test.memory_usage().sum() / 1024**2))
print('X_train Shape = {}'.format(X_train.shape))
print('y_train Shape = {}'.format(y_train.shape))
print('X_test Shape = {}\n'.format(X_test.shape))

DATE_COLS = ['Minute', 'Hour', 'Day', 'DayOfWeek', 'DayOfYear', 'Week', 'Month', 'TransactionDT', 'TransactionDate']

ID_COLS = ['TransactionID', 'Card_ID1', 'Card_ID2', 'Card_ID3', 'Item_ID', 'Device_ID', 'PAccount_ID', 'RAccount_ID', 'PR_emaildomain']

NEW_FEATURES = ['addr2_TransactionAmt_Sum', 'addr2_TransactionAmt_Mean', 'addr2_TransactionAmt_Std', 'addr2_VC', 
                'card4_TransactionAmt_Sum', 'card4_TransactionAmt_Mean', 'card4_TransactionAmt_Std', 'card4_VC', 
                'card6_TransactionAmt_Sum', 'card6_TransactionAmt_Mean', 'card6_TransactionAmt_Std', 'card6_VC']

cols_to_drop = ['V300','V309','V111', 'C3', 'V124','V106','V125','V315','V134','V102','V123','V316','V113',
              'V136','V305','V110','V299','V289','V286','V318','V103','V304','V116','V29','V284','V293',
              'V137','V295','V301','V104','V311','V115','V109','V119','V321','V114','V133','V122','V319',
              'V105','V112','V118','V117','V121','V108','V135','V320','V303','V297','V120'] + DATE_COLS + ID_COLS + NEW_FEATURES

for df in [X_train, X_test]:
    for col in cols_to_drop:
        try:
            df.drop(columns=col, inplace=True)
        except Exception as e:
            print(e, col)

# In[4]:


object_cols = [col for col in X_train.columns if X_train[col].dtype == 'object']
le = LabelEncoder()

for df in [X_train, X_test]:
    for col in object_cols:
        df[col] = le.fit_transform(df[col].astype(str).values)  
    print('{} features are label encoded.'.format(len(object_cols)))


# In[5]:


lgb_param = {
    'min_data_in_leaf': 106,
    'num_leaves': 500,
    'learning_rate': 0.009,
    'min_child_weight': 0.03454472573214212,
    'bagging_fraction': 0.4181193142567742,
    'feature_fraction': 0.3797454081646243,
    'reg_lambda': 0.6485237330340494,
    'reg_alpha': 0.3899927210061127,
    'max_depth': -1,
    'objective': 'binary',
    'seed': SEED,
    'feature_fraction_seed': SEED,
    'bagging_seed': SEED,
    'drop_seed': SEED,
    'data_random_seed': SEED,
    'boosting_type': 'gbdt',
    'verbose': 0,
    'metric':'auc',
}

N = 10
kf = KFold(n_splits=N)
importance = pd.DataFrame(np.zeros((X_train.shape[1], N)), columns=['Fold_{}'.format(i) for i in range(1, N + 1)], index=X_train.columns)
scores = []
y_pred = np.zeros(X_test.shape[0])
oof = np.zeros(X_train.shape[0])

for fold, (trn_idx, val_idx) in enumerate(kf.split(X_train, y_train), 1):
	print('Fold {}'.format(fold))
	trn_data = lgb.Dataset(X_train.iloc[trn_idx, :].values, label=y_train.iloc[trn_idx].values)
	val_data = lgb.Dataset(X_train.iloc[val_idx, :].values, label=y_train.iloc[val_idx].values)
	clf = lgb.train(lgb_param, trn_data, 10000, valid_sets=[trn_data, val_data], verbose_eval=500, early_stopping_rounds=500)
	predictions = clf.predict(X_train.iloc[val_idx, :].values)
	importance.iloc[:, fold - 1] = clf.feature_importance()
	oof[val_idx] = predictions
	score = roc_auc_score(y_train.iloc[val_idx].values, predictions)
	scores.append(score)
	print('Fold {} ROC AUC Score {}\\n'.format(fold, score))
	y_pred += clf.predict(X_test) / N
	del trn_data, val_data, predictions
	gc.collect()
	print('Average ROC AUC Score {} [STD:{}]'.format(np.mean(scores), np.std(scores)))

importance['Mean_Importance'] = importance.sum(axis=1) / N
importance.sort_values(by='Mean_Importance', inplace=True, ascending=False)

print('caching_oov\'s')
np.save()
print('making preds')
submission = pd.read_csv('sample_submission.csv', index_col='TransactionID')
submission['isFraud'] = y_pred
submission.to_csv('submission_lgbm_model_all_feats_16092019.csv')
submission.head()
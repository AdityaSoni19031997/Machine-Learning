
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from time import time
import datetime
import lightgbm as lgb
import gc, warnings
gc.collect()

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, confusion_matrix, accuracy_score
from sklearn.metrics import roc_auc_score, f1_score, roc_curve, auc,precision_recall_curve
from scipy import interp
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


# In[2]:


traintr = pd.read_csv('input/train_transaction.csv.zip')
trainid = pd.read_csv('input/train_identity.csv.zip')
testtr  = pd.read_csv('input/test_transaction.csv.zip')
testid  = pd.read_csv('input/test_identity.csv.zip')


# In[3]:


# For each categorical variable, we'd like to experiment with
# the count of appearances within that day's hour
# This will only work if the distributions (counts) are similar in train + test

START_DATE     = '2017-12-01'
startdate      = datetime.datetime.strptime(START_DATE, '%Y-%m-%d')
traintr['tdt']    = traintr['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds = x)))
traintr['thour']  = traintr.tdt.dt.hour
traintr['tdate']  = traintr.tdt.dt.date

testtr['tdt']    = testtr['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds = x)))
testtr['thour']  = testtr.tdt.dt.hour
testtr['tdate']  = testtr.tdt.dt.date


# In[4]:


data = traintr.append(testtr, sort=False)
data.reset_index(drop=True, inplace=True)
trx_size = traintr.shape[0]


# In[5]:


seed = 123

# Fast FI!
params = {
    # Simple trees
    'max_depth': 16,
    'num_leaves': 4,
#     'bagging_fraction': 0.80,
    
    'objective': 'binary',
    'save_binary': True,
    'seed': seed,
    'feature_fraction_seed': seed,
    'bagging_seed': seed,
    'drop_seed': seed,
    'data_random_seed': seed,
    'boosting_type': 'gbdt',
    'boost_from_average': True,
    'metric':'auc',
    'verbosity': -1,
    'verbose': -1,
    'is_unbalance': False,
    #'scale_pos_weight':2,
}


# This is a very simple method we can use to test covariate shift between train + test. It uses randomforest rather than GBDT. So the results might underestimate the CVS, beware!
# 

# In[ ]:


def test_cvs(data, feature, plot=False):
    global trx_size
    
    data['covariate_shift'] = (np.arange(data.shape[0]) >= trx_size).astype(np.uint8)
    peek = data[~data[feature].isna()]
    
    if plot:
        f, ax = plt.subplots(2,2,figsize=(14,8))
        ax[0, 0].set_title('Train')
        ax[0, 1].set_title('Test')
        ax[0, 0].hist(peek[peek.index<trx_size][feature],100)
        ax[0, 1].hist(peek[peek.index>=trx_size][feature],100)
        # TODO: Plot against TIME:
        ax[1, 0].scatter(peek[peek.index<trx_size].TransactionDT, peek[peek.index<trx_size][feature], s=0.1, alpha=0.1)
        ax[1, 1].scatter(peek[peek.index>=trx_size].TransactionDT, peek[peek.index>=trx_size][feature], s=0.1, alpha=0.1)
        plt.show()
    
    # Test covariate shift:
    index_trn = np.random.choice(np.arange(peek.shape[0]), size=peek.shape[0]//2)
    index_val = list(set(np.arange(peek.shape[0])) - set(index_trn))
    
    clf = RandomForestClassifier(n_estimators=10, max_depth=15, random_state=1237, n_jobs=-1)
    clf.fit(peek[feature].iloc[index_trn].values.reshape(-1,1), peek.covariate_shift.iloc[index_trn]) 

    y_true = peek.iloc[index_val].covariate_shift.values.flatten()
    y_pred = clf.predict(peek[feature].iloc[index_val].values.reshape(-1,1))
    
    del data['covariate_shift']
    return roc_auc_score(y_true, y_pred)


# Simple method allows us to evaluate the RoC of a set of features. Also returns things like gain, splits, etc...

# In[1]:


def experiment(data, features, params, runs=3, train_frac=0.75, seed=123, expnum=0, retoof=False, max_trees=25):
    global trx_size
    
    # Feature Selection Validator
    recalls    = []
    roc_aucs   = []
    f1_scores  = []
    accuracies = []
    precisions = []
    all_y_true = []
    all_y_pred = []
    tprs = []

    fi = pd.DataFrame()
    fi['feature'] = features
    fi['expnum'] = expnum
    
    oof = np.zeros(trx_size) # todo: logic for last fold

    # TODO: Use our validation strategy here rather than % split
    index_trn = np.arange(int(train_frac*trx_size))
    index_val = np.arange(int(train_frac*trx_size),trx_size)
    
    trn_data = lgb.Dataset(data[features].iloc[index_trn], label=data.isFraud.iloc[index_trn])
    val_data = lgb.Dataset(data[features].iloc[index_val], label=data.isFraud.iloc[index_val])

    params = params.copy()
    for run in range(runs):
        params['seed'] = seed + run
        params['feature_fraction_seed'] = seed + run
        params['bagging_seed'] = seed + run
        params['drop_seed'] = seed + run
        params['data_random_seed'] = seed + run

        clf = lgb.train(
            params,
            trn_data,
            max_trees,
            valid_sets = [trn_data, val_data],
            verbose_eval=600,
            early_stopping_rounds=5
        )
        
        # Scores
        y_true = data.iloc[index_val].isFraud.values
        y_pred = clf.predict(data[features].iloc[index_val])
        all_y_true.append(y_true)
        all_y_pred.append(y_pred)
        
        oof[index_val] = y_pred
        
        warnings.filterwarnings("ignore")
        accuracies.append(accuracy_score(y_true, y_pred.round()))
        recalls.append(recall_score(y_true, y_pred.round()))
        precisions.append(precision_score(y_true, y_pred.round()))
        f1_scores.append(f1_score(y_true, y_pred.round()))
        roc_aucs.append(clf.best_score['valid_1']['auc'])
        warnings.filterwarnings("default")
        
        fi['gain_run_{}'.format(run + 1)] = clf.feature_importance(importance_type='gain')
        fi['split_run_{}'.format(run + 1)] = clf.feature_importance(importance_type='split')
        fi['iter_run_{}'.format(run + 1)] = clf.best_iteration
        fi['roc_run_{}'.format(run + 1)] = clf.best_score['valid_1']['auc']

    fi['gain'] = fi[[f for f in fi if 'gain_run_' in f]].mean(axis=1)
    fi['split'] = fi[[f for f in fi if 'split_run_' in f]].mean(axis=1)
    fi['iter'] = fi[[f for f in fi if 'iter_run_' in f]].mean(axis=1)
    fi['roc'] = fi[[f for f in fi if 'roc_run_' in f]].mean(axis=1)

    if retoof:
        return oof, fi, np.mean(roc_aucs)
    
    return fi, np.mean(roc_aucs)


# In[7]:


all_cols = [f for f in data.columns if f not in ['TransactionID', 'TransactionDT', 'isFraud', 'tdt', 'tdate']]
ready_cols = []
for f in all_cols:
    # For string variables, encode to label and count encode.
    if data[f].dtype!='O':
        ready_cols.append(f)
        continue

    mapper = {key:val for val,key in enumerate(data[f].unique())}
    if np.nan in mapper: mapper[np.nan] = np.nan # Keep Nans Nan
        
    # For non-numeric categorical variables, try both Label + CountEncode. Whatever works best..
    data['LE_' + f] = data[f].map(mapper)
    data['CE_' + f] = data[f].map(data[f].value_counts(dropna=True)) # Keep nans nan here as well...
    ready_cols += ['CE_' + f, 'LE_' + f]

len(ready_cols), ready_cols


# In[8]:


trpart = data[data.index<trx_size]
tepart = data[data.index>=trx_size]

results = []
for i, f in enumerate(ready_cols):
    print(i+1, '/', len(ready_cols))

    # CVS by decision tree...
    cvs = test_cvs(data, f)
    
    cnt_nan_train = tepart[f].isna().sum() / traintr.shape[0]
    cnt_nan_test = tepart[f].isna().sum() / testtr.shape[0]
    nunique = data[f].nunique()
    
    vcs = data[f].value_counts()
    most_freq_val  = vcs.index[0]
    most_freq_perc = vcs.iloc[0] / data.shape[0]
    least_freq_cnt = vcs.iloc[-1]
    
    tr = data[data.index<trx_size][[f,'isFraud']]
    corr_tdt = data[[f,'TransactionDT']].corr()[f].TransactionDT
    corr_target = tr.corr()[f].isFraud
    
    exper = experiment(
        tr,
        [f],
        params,
        runs=1,
        train_frac=0.7,
        seed=123,
        expnum=1
    )[0]
    
    results.append([
        f, cvs, nunique,
        cnt_nan_train, cnt_nan_test, most_freq_perc, most_freq_val, least_freq_cnt,
        exper.gain.iloc[0], exper.split.iloc[0], exper.roc.iloc[0], exper.iter.iloc[0], 
        corr_target, corr_tdt
    ])
    
results = pd.DataFrame(results, columns=[
    'col', 'CVS', 'NUnique', 'NANTrainR', 'NANTestR',
    'MFreqR', 'MFreqV', 'LFreqCnt',
    'Gain70','Split70','Roc70','Iter70',
    'YCorr', 'DTCorr'
])
results['GPS70'] = results.Gain70 / results.Split70


# In[9]:


results.sort_values(['Roc70'], ascending=False)


# # Descritize ALL Vars By ValueCount % Bucketing

# In[10]:


# NOTE: ALL values are "numeric" because we've encoded to LE and CE
threshold = 0.02
cutoff = int(threshold * data.shape[0])

for col in ready_cols:
    label = 0
    mapper = {}
    vcs = data[col].value_counts(dropna=False).reset_index()
    for var,cnt in vcs.values:
        mapper[var] = label
        if cnt > cutoff: label += 1
    data[col + '_enc'] = data[col].map(mapper)
    
cnts = data[[v for v in data.columns if '_enc' in v]].nunique()
cnts.sort_values(ascending=False, inplace=True)


# In[11]:


# del results['enc_cnt']

cnts_merge = cnts.reset_index().rename(columns={'index':'col', 0:'enc_cnt'})
cnts_merge.col = cnts_merge.col.apply(lambda x: x[:-4])

results = results.merge(
    cnts_merge,
    how='left', on='col'
)
results


# In[12]:


plt.title('Roc70')
plt.plot(results.Roc70.sort_values(ascending=False).values)
plt.show()

plt.title('CVS')
plt.plot(results.CVS.sort_values(ascending=False).values)
plt.show()

plt.title('NANTestR')
plt.plot(results.NANTestR.sort_values(ascending=False).values)
plt.show()

plt.title('NominalEnc NUnique')
plt.plot(cnts.values)
plt.show()


# In[13]:


candidates = results[
    (results.Roc70 >= 0.6) &
    (results.NANTestR < 0.75) &
    (results.NANTrainR < 0.75) &
    (results.CVS <= 0.525) # perhaps allow this to increase?
].sort_values(['Roc70'], ascending=False)
candidates


# In[14]:


candidates.sort_values('YCorr').tail()


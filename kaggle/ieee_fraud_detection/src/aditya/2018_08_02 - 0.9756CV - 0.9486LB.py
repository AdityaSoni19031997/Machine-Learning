import pandas as pd
import numpy as np
from time import time
import datetime
import lightgbm as lgb
import gc
gc.collect()

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, confusion_matrix, accuracy_score
from sklearn.metrics import roc_auc_score, f1_score, roc_curve, auc,precision_recall_curve
from scipy import interp
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


# In[4]:


traintr = pd.read_csv('input/train_transaction.csv.zip')
trainid = pd.read_csv('input/train_identity.csv.zip')

testtr = pd.read_csv('input/test_transaction.csv.zip')
testid = pd.read_csv('input/test_identity.csv.zip')

nulltr = bothtr[[c for c in bothtr if c != 'isFraud']].isnull().sum().sort_values().reset_index()
nulltr.rename(columns={'index':'col', 0:'nulls'}, inplace=True)
grouped_tr = nulltr.groupby('nulls').col.agg(list).reset_index()

cat = [
    'ProductCD',
    'card1','card2','card3','card4','card5','card6',
    'addr1','addr2',
    'P_emaildomain','R_emaildomain',
    'M1','M2','M3','M4','M5','M6','M7','M8','M9',
]

nunique_col = bothtr[[c for c in bothtr.columns if c not in cat]].nunique()
nunique_col.sort_values().head()

traintr.groupby('V335').isFraud.agg(['size','mean']).sort_values('size', ascending=False)


import logging
from typing import Optional

import numpy as np
from sklearn.model_selection._split import _BaseKFold
from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples

LOGGER = logging.getLogger(__name__)

class TimeSeriesSplit(_BaseKFold):  # pylint: disable=abstract-method
    # https://www.kaggle.com/mpearmain/extended-timeseriessplitter
    """Time Series cross-validator

    Provides train/test indices to split time series data samples that are observed at fixed time intervals,
    in train/test sets. In each split, test indices must be higher than before, and thus shuffling in cross validator is
    inappropriate.

    This cross_validation object is a variation of :class:`TimeSeriesSplit` from the popular scikit-learn package.
    It extends its base functionality to allow for expanding windows, and rolling windows with configurable train and
    test sizes and delays between each. i.e. train on weeks 1-8, skip week 9, predict week 10-11.

    In this implementation we specifically force the test size to be equal across all splits.

    Expanding Window:

            Idx / Time  0..............................................n
            1           |  train  | delay |  test  |                   |
            2           |       train     | delay  |  test  |          |
            ...         |                                              |
            last        |            train            | delay |  test  |

    Rolling Windows:
            Idx / Time  0..............................................n
            1           | train   | delay |  test  |                   |
            2           | step |  train  | delay |  test  |            |
            ...         |                                              |
            last        | step | ... | step |  train  | delay |  test  |

    Parameters:
        n_splits : int, default=5
            Number of splits. Must be at least 4.

        train_size : int, optional
            Size for a single training set.

        test_size : int, optional, must be positive
            Size of a single testing set

        delay : int, default=0, must be positive
            Number of index shifts to make between train and test sets
            e.g,
            delay=0
                TRAIN: [0 1 2 3] TEST: [4]
            delay=1
                TRAIN: [0 1 2 3] TEST: [5]
            delay=2
                TRAIN: [0 1 2 3] TEST: [6]

        force_step_size : int, optional
            Ignore split logic and force the training data to shift by the step size forward for n_splits
            e.g
            TRAIN: [ 0  1  2  3] TEST: [4]
            TRAIN: [ 0  1  2  3  4] TEST: [5]
            TRAIN: [ 0  1  2  3  4  5] TEST: [6]
            TRAIN: [ 0  1  2  3  4  5  6] TEST: [7]
    
    Examples
    --------
    >>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]])
    >>> y = np.array([1, 2, 3, 4, 5, 6])
    >>> tscv = TimeSeriesSplit(n_splits=5)
    >>> print(tscv)  # doctest: +NORMALIZE_WHITESPACE
    TimeSeriesSplit(train_size=None, n_splits=5)
    >>> for train_index, test_index in tscv.split(X):
    ...    print('TRAIN:', train_index, 'TEST:', test_index)
    ...    X_train, X_test = X[train_index], X[test_index]
    ...    y_train, y_test = y[train_index], y[test_index]
    TRAIN: [0] TEST: [1]
    TRAIN: [0 1] TEST: [2]
    TRAIN: [0 1 2] TEST: [3]
    TRAIN: [0 1 2 3] TEST: [4]
    TRAIN: [0 1 2 3 4] TEST: [5]
    """

    def __init__(self,
                 n_splits: Optional[int] = 5,
                 train_size: Optional[int] = None,
                 test_size: Optional[int] = None,
                 delay: int = 0,
                 force_step_size: Optional[int] = None):

        if n_splits and n_splits < 5:
            raise ValueError(f'Cannot have n_splits less than 5 (n_splits={n_splits})')
        super().__init__(n_splits, shuffle=False, random_state=None)

        self.train_size = train_size

        if test_size and test_size < 0:
            raise ValueError(f'Cannot have negative values of test_size (test_size={test_size})')
        self.test_size = test_size

        if delay < 0:
            raise ValueError(f'Cannot have negative values of delay (delay={delay})')
        self.delay = delay

        if force_step_size and force_step_size < 1:
            raise ValueError(f'Cannot have zero or negative values of force_step_size '
                             f'(force_step_size={force_step_size}).')

        self.force_step_size = force_step_size

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.

        Parameters:
            X : array-like, shape (n_samples, n_features)
                Training data, where n_samples is the number of samples  and n_features is the number of features.

            y : array-like, shape (n_samples,)
                Always ignored, exists for compatibility.

            groups : array-like, with shape (n_samples,), optional
                Always ignored, exists for compatibility.

        Yields:
            train : ndarray
                The training set indices for that split.

            test : ndarray
                The testing set indices for that split.
        """
        X, y, groups = indexable(X, y, groups)  # pylint: disable=unbalanced-tuple-unpacking
        n_samples = _num_samples(X)
        n_splits = self.n_splits
        n_folds = n_splits + 1
        delay = self.delay

        if n_folds > n_samples:
            raise ValueError(f'Cannot have number of folds={n_folds} greater than the number of samples: {n_samples}.')

        indices = np.arange(n_samples)
        split_size = n_samples // n_folds

        train_size = self.train_size or split_size * self.n_splits
        test_size = self.test_size or n_samples // n_folds
        full_test = test_size + delay

        if full_test + n_splits > n_samples:
            raise ValueError(f'test_size\\({test_size}\\) + delay\\({delay}\\) = {test_size + delay} + '
                             f'n_splits={n_splits} \n'
                             f' greater than the number of samples: {n_samples}. Cannot create fold logic.')

        # Generate logic for splits.
        # Overwrite fold test_starts ranges if force_step_size is specified.
        if self.force_step_size:
            step_size = self.force_step_size
            final_fold_start = n_samples - (train_size + full_test)
            range_start = (final_fold_start % step_size) + train_size

            test_starts = range(range_start, n_samples, step_size)

        else:
            if not self.train_size:
                step_size = split_size
                range_start = (split_size - full_test) + split_size + (n_samples % n_folds)
            else:
                step_size = (n_samples - (train_size + full_test)) // n_folds
                final_fold_start = n_samples - (train_size + full_test)
                range_start = (final_fold_start - (step_size * (n_splits - 1))) + train_size

            test_starts = range(range_start, n_samples, step_size)

        # Generate data splits.
        for test_start in test_starts:
            idx_start = test_start - train_size if self.train_size is not None else 0
            # Ensure we always return a test set of the same size
            if indices[test_start:test_start + full_test].size < full_test:
                continue
            yield (indices[idx_start:test_start],
                   indices[test_start + delay:test_start + full_test])
            

def examine_feature(feature='P_emaildomain'):
    global traintr, bothtr

    look = bothtr.groupby(feature).size().reset_index()
    look.rename(columns={0:'fullsize'}, inplace=True)
    temp = traintr.groupby(feature).isFraud.agg(['size','mean']).sort_values('mean').reset_index()
    temp.rename(columns={'size':'trsize', 'mean':'isFraud'}, inplace=True)
    look = look.merge(temp, how='left', on=feature)
    
    print(feature, 'num unique vals:', look[feature].nunique())
    return look


def build_ranges(ranges):
    out = []
    for arange in ranges:
        out.append(np.arange(arange[0], arange[-1]+1, 1).tolist())
    return sum(out, [])
    
def target_mean_encode(data, col):
    encode = data.groupby(col).isFraud.mean().sort_values(ascending=False).reset_index()
    mapper = {k:v for v, k in enumerate(encode[col].values)}
    data[col] = data[col].map(mapper)
    return data, mapper

tt = time()
def updateme(msg, reset=False):
    global tt
    if reset: tt = time()
    print(time()-tt, msg)
    tt = time()

def build_features(trx,idn):
    updateme('Mergind DFrame + Computing NANs')
    trx['nulls_trx'] = trx.isna().sum(axis=1)
    idn['nulls_idn'] = idn.isna().sum(axis=1)

    data = trx.merge(idn, how='left', on='TransactionID')
    old_features = [c for c in data.columns if c not in ['nulls_trx', 'nulls_idn']]
    
    # Make sure everything is lowercase
    for c1, c2 in data.dtypes.reset_index().values:
        if not c2=='O': continue
        data[c1] = data[c1].astype(str).apply(str.lower)
    
    updateme('Building Groups')
    stringy = lambda x: x.astype(str) + ' '
    data['CardID'] = stringy(data.card1) + stringy(data.card2) + stringy(data.card3) + stringy(data.card4) + stringy(data.card5) + stringy(data.card6)     + stringy(data.addr1) # + stringy(data.addr2) # Sergey says addr1 only: https://www.kaggle.com/c/ieee-fraud-detection/discussion/101785#latest-588573
    data['DeviceID']  = stringy(data.DeviceType) + stringy(data.DeviceInfo) + stringy(data.id_31) # TODO: Clean
    data['PAccountID'] = stringy(data.addr1) + stringy(data.addr2) + stringy(data.P_emaildomain)
    data['RAccountID'] = stringy(data.addr1) + stringy(data.addr2) + stringy(data.R_emaildomain)

    updateme('Count Encoding Groups')
    # TODO: Try count + label encode (e.g. both)
    for col in ['nulls_idn', 'nulls_trx', 'CardID', 'DeviceID', 'PAccountID', 'RAccountID', 'ProductCD']:
        data[col] = data[col].map(data[col].value_counts(dropna=False))
    
    updateme('Count Encoding Vars')
    count_encode = ['card1', 'id_34', 'id_36', 'TransactionAmt']
    for col in count_encode:
        data['CountEncode_' + col] = data[col].map(data[col].value_counts(dropna=False))
        
        
    updateme('Email Features')
    data['TransactionAmtCents'] = np.ceil(data.TransactionAmt) - np.floor(data.TransactionAmt)
    country_map = {
        'com':'us', 'net':'us', 'edu':'us', 'gmail':'us', 
        'mx': 'mx', 'es':'es', 'de':'de', 'fr':'fr',
        'uk':'uk', 'jp':'jp'
    }
    domain = lambda x: x.split('.')[0]
    pemail_country = lambda x: x.split('.')[-1]
    data['pemail_domain']  = data.P_emaildomain.astype(str).apply(domain)
    data['pemail_ext']     = data.P_emaildomain.astype(str).apply(pemail_country).map(country_map)
    data['remail_domain']  = data.R_emaildomain.astype(str).apply(domain)
    data['remail_ext']     = data.R_emaildomain.astype(str).apply(pemail_country).map(country_map)
    data['p_and_r_email']  = data.P_emaildomain.astype(str) + ' ' + data.R_emaildomain.astype(str)

    updateme('Time Features')
    # We can calculate transaction hour directly;
    # But samples where D9 isna seem to have less fraud rate. And there's a LOT of them:
    data.D9 = data.D9.isnull()
    
    # Time deltas Mod7 and mod(7*4)
    for i in range(1,16):
        if i in [8,9]: continue
        temp = data['D'+str(i)] % 7
        temp.loc[data['D'+str(i)]==0] = -1
        data['D{}_mod7'.format(i)] = temp.values
    
    slope = 1 / (60*60*24) # sec/day
    for i in range(1,16):
        if i in [9]: continue
        feature = 'D' + str(i)
        data[feature+'_mfix'.format(i)] = np.round_(data[feature] - (data.TransactionDT - data.TransactionDT.min()) * slope)
        data[feature+'_mfix_mod7'.format(i)] = data[feature+'_mfix'.format(i)] % 7
        
    START_DATE     = '2017-12-01'
    startdate      = datetime.datetime.strptime(START_DATE, '%Y-%m-%d')
    data['tdt']    = data['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds = x)))
    data['tdow']   = data.tdt.dt.dayofweek
    data['thour']  = data.tdt.dt.hour
    data['tdate']  = data.tdt.dt.date
    
    
    # TODO: Add holidays.
    
    # @9h, id_01 is the least
    # @18h, id_02 is the least
    data['thour_id_01'] = ((np.abs(9 - data.thour) % 12) + 1) * (data.id_01 + 1)
    data['thour_id_02'] = ((np.abs(18 - data.thour) % 12) + 1) * (data.id_02 + 1)
    
    # Groups:
    updateme('Group Aggregates')
    # I'm also trying features like HourTransactionVolume, DayTransactionVolume, etc, but they are not very promising. They tend to increase cv, but decreases lb. I hope this inspires you.
    # temp = data.groupby(['thour','tdate']).size().reset_index()
    # temp.rename(columns={0:'trans_per_hourdate'}, inplace=True)
    # data = data.merge(temp, how='left', on=['thour','tdate'])
    
    temp = data.groupby('thour').size().reset_index()
    temp.rename(columns={0:'trans_per_hour'}, inplace=True)
    data = data.merge(temp, how='left', on='thour')
                  
    cat = 'CardID'
    grp = data.groupby(cat)
    temp = grp.id_02.agg(['min','std'])
    temp.columns = ['G{}_{}_{}'.format(cat, 'id_02', col) for col in ['min','std']]
    data = data.merge(temp, how='left', on=cat)
    
    temp = grp.C13.agg(['std'])
    temp.columns = ['G{}_{}_{}'.format(cat, 'C13', col) for col in ['std']]
    data = data.merge(temp, how='left', on=cat)
    
    temp = grp.TransactionAmt.agg(['max'])
    temp.columns = ['G{}_{}_{}'.format(cat, 'TransactionAmt', col) for col in ['max']]
    data = data.merge(temp, how='left', on=cat)
    
    temp = grp.D1_mfix.agg(['max'])
    temp.columns = ['G{}_{}_{}'.format(cat, 'D1_mfix', col) for col in ['max']]
    data = data.merge(temp, how='left', on=cat)

    cat = 'PAccountID'
    grp = data.groupby(cat)
    temp = grp.dist1.agg(['max', 'std'])
    temp.columns = ['G{}_{}_{}'.format(cat, 'dist1', col) for col in ['max', 'std']]
    data = data.merge(temp, how='left', on=cat)

    cat = 'nulls_trx'
    grp = data.groupby(cat)
    temp = grp.id_02.agg(['max'])
    temp.columns = ['G{}_{}_{}'.format(cat, 'id_02', col) for col in ['max']]
    data = data.merge(temp, how='left', on=cat)
    
    temp = grp.C13.agg(['max'])
    temp.columns = ['G{}_{}_{}'.format(cat, 'C13', col) for col in ['max']]
    data = data.merge(temp, how='left', on=cat)
    
    cat = 'thour'
    temp = data.groupby(cat).TransactionAmt.agg(['min','max','mean','median','std'])
    temp.columns = ['G{}_{}_{}'.format(cat, 'TransactionAmt', col) for col in ['min','max','mean','median','std']]
    data = data.merge(temp, how='left', on=cat)
                    
    cat = 'addr1'
    temp = data.groupby(cat).TransactionAmt.agg(['min','max','mean','median','std'])
    temp.columns = ['G{}_{}_{}'.format(cat, 'TransactionAmt', col) for col in ['min','max','mean','median','std']]
    data = data.merge(temp, how='left', on=cat)
                    
    cat = 'card5'
    temp = data.groupby(cat).TransactionAmt.agg(['min','max','mean','median','std'])
    temp.columns = ['G{}_{}_{}'.format(cat, 'TransactionAmt', col) for col in ['min','max','mean','median','std']]
    data = data.merge(temp, how='left', on=cat)
                    
    # End Groups
    
    # IDEA here is (proven garbage with M5 and D1):
    # Access from outside your country. (IP and browser language settings, time zone) (M? x D? interactions)
    #data['M5_D1_mfix'] = (data.M5.map({'F':2, 'T':1, np.nan:0})+1).astype(np.float) * (data.D1_mfix-data.D1_mfix.min()+1).astype(np.float)
    
    updateme('OHEs...')
    # These just have fun isFraud means
    OHEFeatures = {
        'P_emaildomain': 'protonmail.com',
        'R_emaildomain': 'protonmail.com',
        'card2': 176,
        #'addr2': 65,
        #'V283': 17,
        #'V37': 8,
        #'V45': 4,
    }
    for key, val in OHEFeatures.items(): data['OHE_'+key] = data[key]==val

    # During labeling the categorical values, protonmail.com tends to come up in others. Instead use this as another label. This gained me +0.120.
    
    
    # addr1, addr2 <-- something in there. Also look at dist1 with these
    # dist1 is probably dist from last transaction location
    
    # These guys have the SAME value_count distribution per key as well!
    # V126-V137 looks interesting. maybe a dollar amount or a distance
    # V160-V166 similar to above
    # V202-V206 similar
    # V207-V216 similar
    # V263-V278 similar
    # V306-V321 similar
    # V331-V339 similar
    cols = ['V' + str(col) for col in build_ranges([
        [126,137],
        [160,166],
        [202,216],
        [263,278],
        [306,321],
        [331,339],
    ])]
    
    #traintr['VSUM1'] = traintr.V130+traintr.V133+traintr.V136
    #data['dollar_weirdness'] = data[cols].apply(lambda x: np.unique(x).shape[0], axis=1)
    #data['weirdness'] = data[continuous].apply(lambda x: np.unique(x).shape[0], axis=1)
    
    # V167-V168, V170 has similar distro
    # V153-V158
    
    
#     # Mean value of random columns belonging to samples we misclassify
#     # Pred=0, Actual=1
#     badguy = {
#         'TransactionAmt':    175.132982,
#         'C1':                 26.855532,
#         'C2':                 31.275574,
#         'D1':                 30.312290,
#         'D2':                 91.129977,
#         'V100':                0.107696,
#         'V310':               36.764886,
#     }
#     for key,val in badguy.items():
#         data['BGuy_'+key] = data[key] - val

    # Interaction
    updateme('Interactions')
    interactions = [['addr1','card1'], ['card1','card5'],
                    ['C5','C9'], ['C5','C13'],['C5','C14'],['C13','C14']]
    for a,b in interactions:
        data[a + '_x_' + b] = stringy(data[a]) + stringy(data[b])
        data[a + '_*_' + b] = (data[a] * data[b])
    
    
    del data['tdt'], data['tdate']
    new_features = list(set(data.columns) - set(old_features))
    data.reset_index(drop=True, inplace=True)
    
    return data, new_features


# In[28]:


gc.collect()

trx_size = traintr.shape[0]
trans    = traintr.append(testtr, sort=False)
ids      = trainid.append(testid, sort=False)
trans.reset_index(drop=True, inplace=True)
ids.reset_index(drop=True, inplace=True)

data, new_features = build_features(trans, ids)

new_features   = [f for f in data.columns if f not in ['TransactionDT', 'TransactionID']]
train_features = [f for f in new_features if f not in ['isFraud']]

gc.collect()


# In[35]:


# # TODO: CountEncode appropriate vars
# count_encode = ['CardID', 'p_and_r_email', 'remail_domain', 'pemail_ext', 'remail_ext', 'pemail_domain']
# for col in tqdm(count_encode):
#     data[col] = data[col].map(data[col].value_counts(dropna=False))
    
# LabelEncode
le = LabelEncoder()
le_features = []
for col in tqdm(new_features):
    if data[col].dtype != 'object': continue
        
    # Keep Nans Nan:
    le_features.append(col)
    mapper = {key:val for val,key in enumerate(data[col].unique())}
    if np.nan in mapper: mapper[np.nan] = np.nan
    data[col] = data[col].map(mapper)
    
le_features


# In[29]:


data['noise0'] = np.random.normal(size=data.shape[0])
data['noise1'] = np.random.uniform(size=data.shape[0])
train_features += ['noise0','noise1']


# In[36]:


data_trn = data.iloc[:trx_size]
data_sub = data.iloc[trx_size:]


# In[32]:


# Confusion matrix 
def plot_confusion_matrix(
    cm,
    classes,
    normalize = False,
    title = 'Confusion matrix"',
    cmap = plt.cm.Blues
):
    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 0)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])) :
        plt.text(j, i, cm[i, j],
                 horizontalalignment = 'center',
                 color = 'white' if cm[i, j] > thresh else 'black')
 
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# # Stratified Val

# In[162]:


params = {'num_leaves': 32, #491,
          #'min_child_weight': 0.03454472573214212,
          #'feature_fraction': 0.3797454081646243,
          #'bagging_fraction': 0.4181193142567742,
          #'min_data_in_leaf': 106,
          'objective': 'binary',
          'max_depth': -1,
          'learning_rate': 0.01,
          "boosting_type": "gbdt",
          "bagging_seed": 11,
          "metric": 'auc',
          "verbosity": -1,
          #'reg_alpha': 0.3899927210061127,
          #'reg_lambda': 0.6485237330340494,
          'random_state': 47
}

param_lgb = {
    'bagging_fraction': 0.90,
    'feature_fraction': 0.90,
    'max_depth': 50,
    'min_child_weight': 0.0029805017044362268,
    'min_data_in_leaf': 20,
    'num_leaves': 382,
    'reg_alpha': 1,
    'reg_lambda': 2,

#     'min_data_in_leaf': int(LGB_BO.max['params']['min_data_in_leaf']), 
#     'num_leaves': int(LGB_BO.max['params']['num_leaves']), 
#     #'learning_rate': LGB_BO.max['params']['learning_rate'],
#     'min_child_weight': LGB_BO.max['params']['min_child_weight'],
#     'bagging_fraction': LGB_BO.max['params']['bagging_fraction'], 
#     'feature_fraction': LGB_BO.max['params']['feature_fraction'],
#     'reg_lambda': LGB_BO.max['params']['reg_lambda'],
#     'reg_alpha': LGB_BO.max['params']['reg_alpha'],
#     'max_depth': int(LGB_BO.max['params']['max_depth']), 
    
    'objective': 'binary',
    'save_binary': True,
    'seed': 1337,
    'feature_fraction_seed': 1337,
    'bagging_seed': 1337,
    'drop_seed': 1337,
    'data_random_seed': 1337,
    'boosting_type': 'gbdt',
    'verbose': 1,
    'is_unbalance': False,
    'boost_from_average': True,
    'metric':'auc'
}

features = train_features.copy()
target = 'isFraud'

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold,KFold
# from bayes_opt import BayesianOptimization
from sklearn.metrics import precision_score, recall_score, confusion_matrix, accuracy_score, roc_auc_score, f1_score, roc_curve, auc,precision_recall_curve
from scipy import interp
import itertools

train_df = data_trn.copy()
test_df = data_sub.copy()

nfold = 5
skf = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=42)

oof = np.zeros(len(train_df))
mean_fpr = np.linspace(0,1,100)
cms= []
tprs = []
aucs = []
y_real = []
y_proba = []
recalls = []
roc_aucs = []
f1_scores = []
accuracies = []
precisions = []
predictions = np.zeros(len(test_df))
feature_importance_df = pd.DataFrame()

i = 1
for train_idx, valid_idx in skf.split(train_df, train_df.isFraud.values):
    print("\nfold {}".format(i))
    trn_data = lgb.Dataset(
        train_df.iloc[train_idx][features].values,
        label=train_df.iloc[train_idx][target].values
    )
    val_data = lgb.Dataset(
        train_df.iloc[valid_idx][features].values,
        label=train_df.iloc[valid_idx][target].values
    )
    
    clf = lgb.train(param_lgb, trn_data, num_boost_round = 500, valid_sets = [trn_data, val_data], verbose_eval = 100, early_stopping_rounds = 100)
    oof[valid_idx] = clf.predict(train_df.iloc[valid_idx][features].values) 
    
    predictions += clf.predict(test_df[features]) / nfold
    
    # Scores
    roc_aucs.append(roc_auc_score(train_df.iloc[valid_idx][target].values, oof[valid_idx]))
    accuracies.append(accuracy_score(train_df.iloc[valid_idx][target].values, oof[valid_idx].round()))
    recalls.append(recall_score(train_df.iloc[valid_idx][target].values, oof[valid_idx].round()))
    precisions.append(precision_score(train_df.iloc[valid_idx][target].values ,oof[valid_idx].round()))
    f1_scores.append(f1_score(train_df.iloc[valid_idx][target].values, oof[valid_idx].round()))
    
    # Roc curve by folds
    f = plt.figure(1)
    fpr, tpr, t = roc_curve(train_df.iloc[valid_idx][target].values, oof[valid_idx])
    tprs.append(interp(mean_fpr, fpr, tpr))
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=2, alpha=0.3, label='ROC fold %d (AUC = %0.4f)' % (i,roc_auc))
    
    # Precion recall by folds
    g = plt.figure(2)
    precision, recall, _ = precision_recall_curve(train_df.iloc[valid_idx][target].values, oof[valid_idx])
    y_real.append(train_df.iloc[valid_idx][target].values)
    y_proba.append(oof[valid_idx])
    plt.plot(recall, precision, lw=2, alpha=0.3, label='P|R fold %d' % (i))  
    
    i += 1
    
    # Confusion matrix by folds
    cms.append(confusion_matrix(train_df.iloc[valid_idx][target].values, oof[valid_idx].round()))
    
    # Features imp
    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = features
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = nfold + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

# Metrics
print(
    '\nCV roc score        : {0:.4f}, std: {1:.4f}.'.format(np.mean(roc_aucs), np.std(roc_aucs)),
    '\nCV accuracy score   : {0:.4f}, std: {1:.4f}.'.format(np.mean(accuracies), np.std(accuracies)),
    '\nCV recall score     : {0:.4f}, std: {1:.4f}.'.format(np.mean(recalls), np.std(recalls)),
    '\nCV precision score  : {0:.4f}, std: {1:.4f}.'.format(np.mean(precisions), np.std(precisions)),
    '\nCV f1 score         : {0:.4f}, std: {1:.4f}.'.format(np.mean(f1_scores), np.std(f1_scores))
)

#ROC 
f = plt.figure(1)
plt.plot([0,1],[0,1],linestyle = '--',lw = 2,color = 'grey')
mean_tpr = np.mean(tprs, axis=0)
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, color='blue', label=r'Mean ROC (AUC = %0.4f)' % (np.mean(roc_aucs)),lw=2, alpha=1)

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('LGB ROC curve by folds')
plt.legend(loc="lower right")

# PR plt
g = plt.figure(2)
plt.plot([0,1],[1,0],linestyle = '--',lw = 2,color = 'grey')
y_real = np.concatenate(y_real)
y_proba = np.concatenate(y_proba)
precision, recall, _ = precision_recall_curve(y_real, y_proba)
plt.plot(recall, precision, color='blue', label=r'Mean P|R')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('P|R curve by folds')
plt.legend(loc="lower left")

# Confusion maxtrix & metrics
plt.rcParams["axes.grid"] = False
cm = np.average(cms, axis=0)
class_names = [0,1]
plt.figure()
plot_confusion_matrix(
    cm, 
    classes=class_names, 
    title= 'LGB Confusion matrix [averaged/folds]'
)
plt.show()


# In[182]:


import datetime
START_DATE     = '2017-12-01'
startdate      = datetime.datetime.strptime(START_DATE, '%Y-%m-%d')
traintr['tdt']    = traintr['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds = x)))
traintr['tmonth'] = traintr.tdt.dt.month
print(traintr.tmonth.unique())


testtr['tdt']    = testtr['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds = x)))
testtr['tmonth'] = testtr.tdt.dt.month
print(testtr.tmonth.unique())

train=6mo
offset=1mo
sub=6mo


#A NOTE: we shouldnt use tmonth as a variable.....
# This will give us a repersentative realistic idea of performance
# For final training, we need to scale somehow....
train=2mo, off=1/3mo, sub=2mo


# In[163]:


import numba
from sklearn.metrics import matthews_corrcoef, precision_recall_fscore_support

@numba.jit
def mcc(tp, tn, fp, fn):
    sup = tp * tn - fp * fn
    inf = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if inf==0:
        return 0
    else:
        return sup / np.sqrt(inf)

#@numba.jit
def eval_mcc(y_true, y_prob, show=False):
    """
    A fast implementation of Anokas mcc optimization code.

    This code takes as input probabilities, and selects the threshold that 
    yields the best MCC score. It is efficient enough to be used as a 
    custom evaluation function in xgboost
    
    Source: https://www.kaggle.com/cpmpml/optimizing-probabilities-for-best-mcc
    Source: https://www.kaggle.com/c/bosch-production-line-performance/forums/t/22917/optimising-probabilities-binary-prediction-script
    Creator: CPMP
    """
    idx = np.argsort(y_prob)
    y_true_sort = y_true[idx]
    n = y_true.shape[0]
    nump = 1.0 * np.sum(y_true) # number of positive
    numn = n - nump # number of negative
    tp = nump
    tn = 0.0
    fp = numn
    fn = 0.0
    best_mcc = 0.0
    best_id = -1
    prev_proba = -1
    best_proba = -1
    mccs = np.zeros(n)
    for i in range(n):
        # all items with idx < i are predicted negative while others are predicted positive
        # only evaluate mcc when probability changes
        proba = y_prob[idx[i]]
        if proba != prev_proba:
            prev_proba = proba
            new_mcc = mcc(tp, tn, fp, fn)
            if new_mcc >= best_mcc:
                best_mcc = new_mcc
                best_id = i
                best_proba = proba
        mccs[i] = new_mcc
        if y_true_sort[i] == 1:
            tp -= 1.0
            fn += 1.0
        else:
            fp -= 1.0
            tn += 1.0
    if show:
        y_pred = (y_prob >= best_proba).astype(int)
        score = matthews_corrcoef(y_true, y_pred)
        plt.plot(mccs)
        return best_proba, best_mcc, y_pred
    else:
        return best_proba,  best_mcc, None

bp, bmcc, yp = eval_mcc(
    y_true = train_df[target].values,
    y_prob = oof,
    show=True
)


# In[164]:


bp, bmcc


# In[165]:


# Improvement Area: Pred==0, Actual==1
pred = oof>bp
examine = traintr[pred==False]
examine = examine[examine.isFraud==1]
examine.shape


# In[166]:


badguy = examine.mean()
badguy[['TransactionAmt', 'C1', 'C2', 'D1', 'D2', 'V100', 'V310']]
badguy = {
    'TransactionAmt':    175.132982,
    'C1':                 26.855532,
    'C2':                 31.275574,
    'D1':                 30.312290,
    'D2':                 91.129977,
    'V100':                0.107696,
    'V310':               36.764886,
}
# pick a few columns:


# In[167]:


import seaborn as sns
plt.style.use('dark_background')
cols = (feature_importance_df[["Feature", "importance"]]
    .groupby("Feature")
    .mean()
    .sort_values(by="importance", ascending=False)[:30].index)
best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)]

plt.figure(figsize=(10,10))
sns.barplot(x="importance", y="Feature", data=best_features.sort_values(by="importance",ascending=False),
        edgecolor=('white'), linewidth=2, palette="rocket")
plt.title('LGB Features importance (averaged/folds)', fontsize=18)
plt.tight_layout()


# In[113]:


sample_submission = pd.read_csv('./input/sample_submission.csv.zip')
sample_submission['isFraud'] = predictions
sample_submission.to_csv('sub.csv', index=False)
get_ipython().system('zip -r -X sub.zip sub.csv')


# In[15]:


sample_submission.head()


# In[169]:


look = feature_importance_df.groupby('Feature').importance.mean().sort_values(ascending=False)
look[50:100]


# # Better time split

# In[170]:


seed = 123

params = {
    'bagging_fraction': 0.80,
    'feature_fraction': 0.3,
    'max_depth': 50,
    'min_child_weight': 0.00298,
    'min_data_in_leaf': 20,
    'num_leaves': 382,
    'reg_alpha': 1,
    'reg_lambda': 2,

#     'min_data_in_leaf': int(LGB_BO.max['params']['min_data_in_leaf']), 
#     'num_leaves': int(LGB_BO.max['params']['num_leaves']), 
#     #'learning_rate': LGB_BO.max['params']['learning_rate'],
#     'min_child_weight': LGB_BO.max['params']['min_child_weight'],
#     'bagging_fraction': LGB_BO.max['params']['bagging_fraction'], 
#     'feature_fraction': LGB_BO.max['params']['feature_fraction'],
#     'reg_lambda': LGB_BO.max['params']['reg_lambda'],
#     'reg_alpha': LGB_BO.max['params']['reg_alpha'],
#     'max_depth': int(LGB_BO.max['params']['max_depth']), 
    
    'objective': 'binary',
    'save_binary': True,
    'seed': seed,
    'feature_fraction_seed': seed,
    'bagging_seed': seed,
    'drop_seed': seed,
    'data_random_seed': seed,
    'boosting_type': 'gbdt',
    'verbose': 1,
    'boost_from_average': True,
    'metric':'auc',
    
    'is_unbalance': False,
    #'scale_pos_weight':2,
}


# In[104]:


# Who are these guys?
# Can we transform them into a single meta-feature and avoid overfitting somehow:
# train_features = [f for f in train_features if f not in ['V188', 'V189', 'V200', 'V201', 'V244', 'V246', 'V257', 'V258']]
train_features = [f for f in train_features if f not in [
    'C7',
    'V45',
    'V186', 'V187', 'V188', 'V189', 'V190',
    'V199', 'V200', 'V201',
    'V242', 'V243', 'V244', 'V245', 'V246',
    'V257', 'V258', 'V259',
]]


# In[131]:


# Who are these guys?
# Can we transform them into a single meta-feature and avoid overfitting somehow:
# train_features = [f for f in train_features if f not in ['V188', 'V189', 'V200', 'V201', 'V244', 'V246', 'V257', 'V258']]
likely_larbage = [
    'OHE_R_emaildomain',
    'V305',
    'OHE_P_emaildomain',
    'id_27',
    'V65',
    'V68',
    'V120',
    'V241',
    'V240',
    'V107',
    'V122',
    'V117',
    'V118',
    'V121'
]

# train_features = [f for f in train_features if f not in [
# #     'C7', 'C11', 'C12', #2
#     'C4',
#     'V44', 'V45',
#     'V186', 'V187', 'V188', 'V189', 'V190',
#     'V199', 'V200', 'V201',
#     'V242', 'V243', 'V244', 'V245', 'V246',
#     'V257', 'V258', 'V259',
# ]]#  and f not in likely_larbage]


# In[179]:


opt = pd.DataFrame({
    'train_frac': [0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.4, 0.4, 0.5, 0.5, 0.6, 0.6, 0.7, 0.7, 0.8, 0.8, 0.9, 0.9],
    'n_splits': [73, 85, 47, 61, 129, 148, 210, 202, 238, 232, 245, 251, 253, 241, 308, 309, 100, 83],
    'rowauc': [0.8690, 0.8721, 0.873646, 0.868276, 0.886307, 0.885623, 0.905196, 0.904122, 0.910311, 0.911545, 0.920712, 0.921639, 0.92853, 0.928586, 0.933343, 0.934491, 0.943982, 0.943013],
})
opt = opt.groupby('train_frac').mean()

plt.plot(opt.n_splits)
ax = plt.twinx()
ax.plot(opt.rowauc, c='orange', alpha=0.5)
plt.show()

opt


# In[169]:


# Feature Selection Validator
recalls    = []
roc_aucs   = []
f1_scores  = []
accuracies = []
precisions = []
all_y_true = []
all_y_pred = []
tprs = []

FPs = []
FNs = []
y_indices = []

# preds = np.zeros(len(test))
oof = np.zeros(trx_size) # todo: logic for last fold
mean_fpr = np.linspace(0,1,100)
fi = pd.DataFrame()
fi['feature'] = train_features

one5 = data_trn.shape[0]//3
tscv = TimeSeriesSplit(train_size=one5, test_size=one5, force_step_size=one5//2, n_splits=5)

f, axs = plt.subplots(1,2,figsize=(15,6))
axs[0].set_title('LGB ROC curve by folds')
axs[0].set_xlabel('False Positive Rate')
axs[0].set_ylabel('True Positive Rate')
axs[1].set_title('P|R curve by folds')
axs[1].set_xlabel('Recall')
axs[1].set_ylabel('Precision')

HOLDA = [
#     [np.arange(trx_size//2), np.arange(trx_size//2,trx_size)], #238
    [np.arange(9*trx_size//10), np.arange(9*trx_size//10,trx_size)], #238
]

training_start_time = time()
# for fold, (index_trn, index_val) in enumerate(tscv.split(data_trn)):
for fold, (index_trn, index_val) in enumerate(HOLDA):
    start_time = time()
    print('Training on fold {}'.format(fold + 1))
    print('TRAIN:', len(index_trn), 'TEST:', len(index_val))
    print(index_trn, index_val)
    
    trn_data = lgb.Dataset(data_trn[train_features].iloc[index_trn], label=data_trn.isFraud.iloc[index_trn])
    val_data = lgb.Dataset(data_trn[train_features].iloc[index_val], label=data_trn.isFraud.iloc[index_val])
    
    clf = lgb.train(
        params,
        trn_data,
        10000,
        valid_sets = [trn_data, val_data],
        verbose_eval=600,
        early_stopping_rounds=200
    )
    
    
    #preds += clf.predict(test)
    pred = clf.predict(data_trn[train_features].iloc[index_val])
    oof[index_val] = pred
    y_indices.append(index_val)

    # Scores
    # TODO: Calc best threshold???
    y_true = data_trn.iloc[index_val].isFraud.values
    y_pred = oof[index_val]
    all_y_true.append(y_true)
    all_y_pred.append(y_pred)
    
    accuracies.append(accuracy_score(y_true, y_pred.round()))
    recalls.append(recall_score(y_true, y_pred.round()))
    precisions.append(precision_score(y_true, y_pred.round()))
    f1_scores.append(f1_score(y_true, y_pred.round()))
    roc_aucs.append(clf.best_score['valid_1']['auc'])
    
    # Roc, Precion, and Recall curves by fold:
    fpr, tpr, _ = roc_curve(y_true, y_pred) # I believe also returns thresh
    tprs.append(interp(mean_fpr, fpr, tpr))
    axs[0].plot(fpr, tpr, lw=2, alpha=0.3, label='ROC fold %d (AUC = %0.4f)' % (fold+1, roc_aucs[-1]))
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    axs[1].plot(recall, precision, lw=2, alpha=0.3, label='P|R fold %d' % (fold+1))  
    
    fi['gain_fold_{}'.format(fold + 1)] = clf.feature_importance(importance_type='gain')
    fi['split_fold_{}'.format(fold + 1)] = clf.feature_importance(importance_type='split')
    print('Fold {} finished in {}'.format(fold + 1, str(datetime.timedelta(seconds=time() - start_time))), end='\n\n')
    
    
print('-' * 30)
print('Training has finished.')
print('Total training time is {}'.format(str(datetime.timedelta(seconds=time() - training_start_time))))
print(
    '\n\tCV roc score        : {0:.4f}, std: {1:.4f}.'.format(np.mean(roc_aucs), np.std(roc_aucs)),
    '\n\tCV accuracy score   : {0:.4f}, std: {1:.4f}.'.format(np.mean(accuracies), np.std(accuracies)),
    '\n\tCV recall score     : {0:.4f}, std: {1:.4f}.'.format(np.mean(recalls), np.std(recalls)),
    '\n\tCV precision score  : {0:.4f}, std: {1:.4f}.'.format(np.mean(precisions), np.std(precisions)),
    '\n\tCV f1 score         : {0:.4f}, std: {1:.4f}.'.format(np.mean(f1_scores), np.std(f1_scores))
)
print('-' * 30)


#ROC 
axs[0].legend(loc="lower right")
axs[0].plot([0,1],[0,1],linestyle = '--',lw = 2,color = 'grey')
mean_tpr = np.mean(tprs, axis=0)
mean_auc = auc(mean_fpr, mean_tpr)
axs[0].plot(mean_fpr, mean_tpr, color='blue', label=r'Mean ROC (AUC = %0.4f)' % (np.mean(roc_aucs)),lw=2, alpha=1)

# PR plt
axs[1].legend(loc="lower left")
axs[1].plot([0,1],[1,0],linestyle = '--',lw = 2,color = 'grey')
all_y_true = np.concatenate(all_y_true)
all_y_pred = np.concatenate(all_y_pred)
precision, recall, _ = precision_recall_curve(all_y_true, all_y_pred)
axs[1].plot(recall, precision, color='blue', label=r'Mean P|R')
plt.show()


# FIs:
fi['gain'] = fi[[f for f in fi if 'gain_fold_' in f]].mean(axis=1)
fi['split'] = fi[[f for f in fi if 'split_fold_' in f]].mean(axis=1)
fi.to_csv('feature_importances.csv')
cols = fi.sort_values(by="gain", ascending=False).feature[:50]
best_features = fi.loc[fi.feature.isin(cols)].sort_values(by='gain', ascending=False)

plt.figure(figsize=(16, 16))
sns.barplot(x="gain", y="feature", data=best_features, edgecolor=('white'), linewidth=2)#, palette="rocket")
plt.title('LGB Features importance (averaged/folds)', fontsize=18)
plt.tight_layout()

CV roc score        : 0.9108, std: 0.0000. 
	CV accuracy score   : 0.9732, std: 0.0000. 
	CV recall score     : 0.3133, std: 0.0000. 
	CV precision score  : 0.8651, std: 0.0000. 
	CV f1 score         : 0.4600, std: 0.0000.
# In[129]:


fi['gps'] = fi.gain / fi.split
z = fi[['feature','gain','split','gps']].sort_values(by='gps', ascending=False).reset_index(drop=True)
z


# In[108]:


z = fi.sort_values(by='gain', ascending=False).reset_index(drop=True)
z


# In[109]:


# TODO: Get rid of some of these
z[z.feature.str.contains('_x_|_\*_')]


# In[110]:


plt.plot(z.gain.values)
plt.show()
for i in range(0,600,50):
    print(z[i:i+50][['feature','gain']], end='\n\n')


# # SUBMISSION

# In[201]:


# Feature Selection Validator

preds = []

training_start_time = time()
# for fold, (index_trn, index_val) in enumerate(tscv.split(data_trn)):
for run in range(3):
    start_time = time()
    print('Training on run {}'.format(run + 1))

    trn_data = lgb.Dataset(data_trn[train_features], label=data_trn.isFraud)
    
    params['seed']= seed+run
    params['feature_fraction_seed']= seed+run
    params['bagging_seed']= seed+run
    params['drop_seed']= seed+run
    params['data_random_seed']= seed+run
    
    clf = lgb.train(
        params,
        trn_data,
        valid_sets = [trn_data],
        num_boost_round=265,
        early_stopping_rounds=2000
    )
    
    preds.append( clf.predict(data_sub[train_features]) )
    print('Fold {} finished in {}'.format(fold + 1, str(datetime.timedelta(seconds=time() - start_time))), end='\n\n')
    


# In[202]:


import numpy as np
p = np.array(preds)
plt.hist(p.std(axis=0), 100)
plt.show()

sample_submission = pd.read_csv('./input/sample_submission.csv.zip')
sample_submission['isFraud'] = p.mean(axis=0)
sample_submission.to_csv('sub.csv', index=False)
get_ipython().system('zip -r -X sub.zip sub.csv')


# # More EDA

# In[ ]:


V160 = grand sum
get_ipython().set_next_input('V165 = half sum');get_ipython().run_line_magic('pinfo', 'sum')
V332,333,203 = sums


# In[407]:


import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# In[386]:


# TransactionAmt
# C10
# V126
# V202
# V208
# V210
# V263
# V306
# V331

#V313 D3
traintr.iloc[21651]


# In[335]:


peak = traintr[new_features+['isFraud']].groupby('dollar_weirdness').isFraud.agg(['size','mean']).reset_index().sort_values('dollar_weirdness', ascending=False)
plt.axhline(y=traintr.isFraud.mean(), linestyle='--', linewidth=1, color='red')
plt.scatter(peak.dollar_weirdness.values, peak['mean'].values, s=1)
plt.show()


# In[317]:


# Look at the unique values in each col to see if we can do simple FE
for col in traintr.columns:
    look = traintr[col].value_counts().sort_values(ascending=False)
    print(col)
    print(look.head(15), end='\n\n\n')


# # EDA Look at each col that has > maxbin unique vals for groupings to extract:

# In[421]:


vcs = [[col,traintr[col].nunique()] for col in traintr.columns]
vcs = pd.DataFrame(vcs, columns=['col','counts'])
vcs.sort_values('counts', ascending=False)
examine = vcs[vcs.counts<256].col
examine


# In[428]:


importantvar = traintr.groupby('V258').isFraud.agg(['size','mean']).sort_values('mean',ascending=False).reset_index()
plt.scatter(importantvar.V258, importantvar['mean'].values, s=importantvar.size, alpha=0.25)
plt.show()
importantvar


# In[50]:


def timeplot(data, f):
    _, ax = plt.subplots(1,2,figsize=(12,7))
    z = traintr[['TransactionDT',f,'isFraud']].copy()
    z.replace([np.inf, -np.inf], np.nan, inplace=True)
    z = z[~z[f].isna()]
    ax[0].set_title(f + ' Time Series')
    ax[0].scatter(z[z.isFraud==0].TransactionDT, z[z.isFraud==0][f],s=1,alpha=0.1)
    ax[0].scatter(z[z.isFraud==1].TransactionDT, z[z.isFraud==1][f],s=1,alpha=0.1)
    ax[1].set_title(f + ' Histogram')
    ax[1].hist(
        [
            z[z.isFraud==0][f],
            z[z.isFraud==1][f]
        ],
        100, #z[f].nunique(),
        stacked=True, density=True, orientation='horizontal'
    )
    ax[1].axes.get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.show()


# In[797]:


timeplot(traintr, 'addr1')
timeplot(traintr, 'addr2')


# # Test for good OHE Features:

# In[105]:


# Find values in columns where isFraud/isFraudShape >> !isFraud/~isFraudShape
min_fraud_diff = 0.3
mean_isFraud = traintr.isFraud.mean()

OHE = {}
for col in tqdm(traintr.columns):
    if col in ['isFraud', 'TransactionID']: continue # -_-!

    col_nu = traintr[col].nunique()
    if col_nu < 45: continue # Number val bounds
    if col_nu > 255*2: continue # Number val bounds
        
    frate = traintr.groupby(col).isFraud.agg(['size','mean'])
    frate = frate[frate['size']>50].reset_index().sort_values('mean', ascending=False) # Sample Bounds
    if frate.shape[0] == 0: continue

    OHE[col] = {
        'na_ratio': np.round(100*traintr[col].isna().sum() / traintr.shape[0],3),
        'nunique': col_nu,
    }
    for val, cnt, fmean in frate.values:
        diff = fmean - mean_isFraud
        # We Focus on Target==1 rather than 0 due to data imbalance...
        # if np.abs(diff) < min_fraud_diff: continue
        if diff < min_fraud_diff: continue
        OHE[col][val] = [int(cnt), np.round(diff,5), ]#, np.round(fmean,5)]
        # 'protonmail.com': [76, 0.3729, 0.40789]},
        
    if len(OHE[col]) == 2:
        del OHE[col]
        
OHE


# In[86]:


import datetime
START_DATE = '2017-12-01'
startdate = datetime.datetime.strptime(START_DATE, '%Y-%m-%d')
traintr['tdt'] = traintr['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds = x)))


# In[98]:


plt.plot(traintr.rolling(on='tdt',window=datetime.timedelta(days = 7)).isFraud.mean(), c='blue')
plt.show()


# In[103]:


plt.title('Num trx per month')
plt.hist(traintr.tdt,182)
plt.show()


# In[113]:


plt.title('Num fraud per month')
z = traintr[traintr.isFraud==1]
z = z[z.tdt<'2017-12-07']
plt.hist(z.tdt,7*24) # days
plt.show()


# In[107]:


traintr['tdt_d'] = traintr.tdt.dt.date
z = traintr[traintr.isFraud==1].groupby('tdt_d').size().sort_values(ascending=False)
plt.plot(z.values)
plt.show()

def timeplot(data, f):
    _, ax = plt.subplots(1,2,figsize=(12,7))
    z = traintr[['TransactionDT',f,'isFraud']].copy()
    z.replace([np.inf, -np.inf], np.nan, inplace=True)
    z = z[~z[f].isna()]
    ax[0].set_title(f + ' Time Series')
    ax[0].scatter(z[z.isFraud==0].TransactionDT, z[z.isFraud==0][f],s=1,alpha=0.1)
    ax[0].scatter(z[z.isFraud==1].TransactionDT, z[z.isFraud==1][f],s=1,alpha=0.1)

    ax[1].set_title(f + ' Histogram')
    ax[1].hist(
        [
            z[z.isFraud==0][f],
            z[z.isFraud==1][f]
        ],
        100, #z[f].nunique(),
        stacked=True, density=True, orientation='horizontal'
    )
    ax[1].axes.get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.show()


# In[ ]:


# for c in range(1,15):
#     C = 'C'+str(c)
#     for d in range(1,16):
#         D = 'D'+str(d)
#         traintr['look'] = traintr[C] / (1 + (traintr[D] %7))
#         print(C, '/', D, 'mod7+1',traintr[D].mean(), traintr[C].mean())
#         timeplot(traintr, 'look')
#         print('')


# In[59]:


traintr.loc[:,traintr.columns[traintr.columns.str.startswith('D')]].isnull().sum().sort_values()


# In[78]:


# # calc mean and median D* value per month
# # store difference between ours and the mean or median

# Note: id vars (and D) are the only ones that change f(x)
# So we have to group on other vars instead...


# import datetime
# START_DATE     = '2017-12-01'
# startdate      = datetime.datetime.strptime(START_DATE, '%Y-%m-%d')
# traintr['tdt']    = traintr['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds = x)))
# traintr['tmonth'] = traintr.tdt.dt.month
# traintr['tweek'] = traintr.tdt.dt.week

var = 'V20'
grp = traintr[[var,'tweek']].groupby('tweek')[var]
mean = grp.mean().reset_index()
median = grp.median().reset_index()
mean.rename(columns={var:var+'_weeklymean'}, inplace=True)
median.rename(columns={var:var+'_weeklymedian'}, inplace=True)

traintr = traintr.merge(mean, how='left', on='tweek')
traintr = traintr.merge(median, how='left', on='tweek')
traintr[var+'_suba'] = traintr[var] - traintr[var+'_weeklymean']
traintr[var+'_subb'] = traintr[var] - traintr[var+'_weeklymedian']

timeplot(traintr, var+'_suba')
timeplot(traintr, var+'_subb')


# In[85]:


z = traintr[traintr.isFraud==1]
timeplot(z, 'card3')
timeplot(z, 'card5')


# In[129]:


traintr = traintr.merge(trainid[['TransactionID','id_01']], how='left', on='TransactionID')


# In[123]:


z = trainid.id_01.value_counts().sort_index().reset_index()
a = z.copy()
a['index'] += 105
z = z.append(a, sort=False)
# z = z.iloc[:74]
plt.plot(z['index'], z.id_01)
plt.show()

id_01 is continuous
thour is continuous
id_01 exhibit bimodal distribution
use cos or sin to encode over its period synced with transaction dt

(within12(thour+thour_off) / % 12) * id_01


# In[238]:


numdays = 5
offhours = 10

traintr['off_tdt'] = traintr.TransactionDT + (24-offhours)*3600
minn = traintr.off_tdt.min() - (traintr.off_tdt.min() % (3600*24))
#print('minn', minn, traintr.off_tdt.min())
traintr.off_tdt -= minn
minn=0

plt.figure(figsize=(16,10))
for i in range(10): plt.axvline(x=3600*24*i, c='orange', alpha=0.5)
for i in range(10): plt.axvline(x=3600*12 + 3600*24*i, c='red', alpha=0.25)
# plt.scatter(traintr.TransactionDT, traintr.id_01, s=.05)
plt.plot(traintr.off_tdt, traintr.id_01, alpha=0.4)
plt.xlim(minn, minn + 3600*24*numdays)
plt.show()


# orange = np.floor(off_tdt / (3600*24)) * (3600*24)
# red = orange + 3600*12

# 'tdt_id_01' = (red-(tdt+3600*offhours)) * id_01
# 'tdt_id_02' = (red-(tdt+3600*offhours)) * id_02


# In[150]:


plt.scatter(traintr.TransactionDT, traintr.id_02, s=.1)
plt.xlim(traintr.TransactionDT.min(), traintr.TransactionDT.min() + 3600*24*5)
plt.show()


# In[261]:


plt.scatter(traintr.thour, traintr.id_01, s=.01)
plt.show()

plt.scatter(traintr.thour, traintr.id_02, s=.01)
plt.show()


# In[ ]:


del traintr['thour_id_01_sum']


# In[269]:


traintr['thour'] = np.floor(traintr.TransactionDT / 3600) % 24
# look = traintr.groupby('thour').id_01
# plt.plot(look.mean())
# ax = plt.twinx()
# ax.plot(look.sum(), c='orange')
# plt.show()

look = traintr.groupby('thour').id_01.mean().reset_index()
look.rename(columns={'id_01':'thour_id_01_mean'}, inplace=True)
traintr = traintr.merge(look, how='left', on='thour')
traintr['thour_id_01_mean_diff'] = traintr.id_01 - traintr.thour_id_01_mean # or std or error or ...
# @ 9h, id_01 is the least
# @18h, id_02 is the least


# In[271]:


offhours = 10

minn = traintr.TransactionDT.min() + 3600*24*40
numdays = 5
offhours = 10
plt.figure(figsize=(16,10))
for i in range(10): plt.axvline(x=minn + 3600*offhours      + 3600*24*i, c='orange', alpha=0.5)
for i in range(10): plt.axvline(x=minn + 3600*(offhours+12) + 3600*24*i, c='red', alpha=0.25)
plt.scatter(traintr[traintr.isFraud==0].TransactionDT, traintr[traintr.isFraud==0].thour_id_01_mean_diff, s=.05)
plt.scatter(traintr[traintr.isFraud==1].TransactionDT, traintr[traintr.isFraud==1].thour_id_01_mean_diff, s=.05)

# plt.plot(traintr.TransactionDT, traintr.id_01, alpha=0.4)
# plt.xlim(minn, minn + 3600*24*numdays)
plt.show()


# # Modeling Time

# In[302]:


traintr[z==True].card3.value_counts()


# # SNS EDA

# In[6]:


# Sample 500 fraud and 500 non-fraud examples to plot
sampled_train = pd.concat([
    traintr.loc[traintr['isFraud'] == 0].sample(5000),
    traintr.loc[traintr['isFraud'] == 1].sample(5000)
])


# In[15]:


c_cols = ['C'+str(i) for i in range(1,15)]


# In[17]:


sns.pairplot(
    sampled_train[['isFraud']+c_cols], 
    hue='isFraud',
    vars=c_cols,
    #plot_kws=dict(s=50, edgecolor="b", linewidth=1)
    plot_kws=dict(s=10, alpha=0.2)
)
plt.show()


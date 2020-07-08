
# coding: utf-8

# In[853]:


# for  C4, C6, C7, C8, C10 outliers, lookit cat variables to see if we can identify groupings...
#C8,c10 we can kinda tell, 0.51
# C12=0.553

# Hard winsorize:
traintr.loc[traintr.D4>484,'D4'] = 485
testtr.loc[testtr.D4>484,'D4'] = 485
data.loc[data.D4>484,'D4'] = np.nan
test_cvs(data, 'D4')

traintr['look'] = traintr.C1 + traintr.C2 + traintr.C11
testtr['look'] = testtr.C1 + testtr.C2 + testtr.C11

START_DATE     = '2017-12-01'
startdate      = datetime.datetime.strptime(START_DATE, '%Y-%m-%d')
traintr['tdt']    = traintr['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds = x)))
traintr['tmonth']   = traintr.tdt.dt.month


import pandas as pd
import numpy as np
from time import time
import datetime
import lightgbm as lgb
import gc, warnings
gc.collect()

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, confusion_matrix, accuracy_score
from sklearn.metrics import roc_auc_score, f1_score, roc_curve, auc,precision_recall_curve
from scipy import interp
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# In[278]:


id_30_dates = {
    # https://en.wikipedia.org/wiki/Android_version_history
    'Android 4.4.2':'2012-11-13',
    'Android 5.0':'2014-11-12',
    'Android 5.0.2':'2014-12-19',
    'Android 5.1.1':'2015-04-21',
    'Android 6.0':'2015-10-05',
    'Android 6.0.1':'2015-12-07',
    'Android 7.0':'2016-08-22',
    'Android 7.1.1':'2016-12-05',
    'Android 7.1.2':'2017-04-04',
    'Android 8.0.0':'2017-08-21',
    'Android 8.1.0':'2017-12-05',
    'Android 9':'2018-08-06',

    'Windows XP':'2001-10-25',
    'Windows Vista':'2006-11-08',
    'Windows 7':'2009-10-22',
    'Windows 8':'2012-10-26',
    'Windows 8.1':'2013-10-17',
    'Windows 10':'2015-07-29',

    # https://robservatory.com/a-useless-analysis-of-os-x-release-dates/
    'Mac OS X 10.6': '2009-08-28',
    'Mac OS X 10_6_8': '2011-06-23',
    'Mac OS X 10_7_5': '2012-09-19',
    'Mac OS X 10_8_5': '2013-09-12',
    'Mac OS X 10.9': '2013-10-22',
    'Mac OS X 10_9_5': '2014-09-17',
    'Mac OS X 10.10': '2014-10-16',
    'Mac OS X 10_10_5': '2015-08-13',
    'Mac OS X 10.11': '2015-09-30',
    'Mac OS X 10_11_3': '2016-01-19',
    'Mac OS X 10_11_4': '2016-03-20',
    'Mac OS X 10_11_5': '2016-05-16',
    'Mac OS X 10_11_6': '2016-07-18',
    'Mac OS X 10.12': '2016-09-20',
    'Mac OS X 10_12': '2016-09-20',
    'Mac OS X 10_12_1': '2016-10-24',
    'Mac OS X 10_12_2': '2016-12-13',
    'Mac OS X 10_12_3': '2017-01-23',
    'Mac OS X 10_12_4': '2017-03-27',
    'Mac OS X 10_12_5': '2017-05-15',
    'Mac OS X 10_12_6': '2017-07-19',
    'Mac OS X 10.13': '2017-09-25',
    'Mac OS X 10_13_1': '2017-10-31',
    'Mac OS X 10_13_2': '2017-12-06',
    'Mac OS X 10_13_3': '2018-01-23',
    'Mac OS X 10_13_4': '2018-03-29',
    'Mac OS X 10_13_5': '2018-06-01',
    'Mac OS X 10_13_6': '2018-07-09',
    'Mac OS X 10.14': '2018-09-24',
    'Mac OS X 10_14': '2018-09-24',
    'Mac OS X 10_14_0': '2018-09-24',
    'Mac OS X 10_14_1': '2018-10-30',
    'Mac OS X 10_14_2': '2018-12-05',

    'iOS 9.3.5':'2016-08-25',
    'iOS 10.0.2':'2016-09-23',
    'iOS 10.1.1':'2016-10-31',
    'iOS 10.2.0':'2016-12-12',
    'iOS 10.2.1':'2017-01-23',
    'iOS 10.3.1':'2017-04-03',
    'iOS 10.3.2':'2017-05-15',
    'iOS 10.3.3':'2017-07-19',
    'iOS 11.0.0':'2017-08-19',
    'iOS 11.0.1':'2017-08-26',
    'iOS 11.0.2':'2017-10-03',
    'iOS 11.0.3':'2017-10-11',
    'iOS 11.1.0':'2017-10-31',
    'iOS 11.1.1':'2017-11-08',
    'iOS 11.1.2':'2017-11-16',
    'iOS 11.2.0':'2017-12-02',
    'iOS 11.2.1':'2017-12-13',
    'iOS 11.2.2':'2018-01-08',
    'iOS 11.2.5':'2018-01-23', 
    'iOS 11.2.6':'2018-02-19',
    'iOS 11.3.0':'2018-03-29',
    'iOS 11.3.1':'2018-04-24',
    'iOS 11.4.0':'2018-05-29',
    'iOS 11.4.1':'2018-07-09',
    'iOS 12.0.0':'2018-08-17',
    'iOS 12.0.1':'2018-09-08',
    'iOS 12.1.0':'2018-09-30',
    'iOS 12.1.1':'2018-12-05',
    'iOS 12.1.2':'2018-12-20',
}

id_30_dates = {k.lower():v for k,v in id_30_dates.items()}


# # Various FE

# In[2]:


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


# In[287]:


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


# # Categorical FE

# In[336]:


def build_cat_features(trx, idn):
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
    
    updateme('Breaking Groups')
    data.loc[
        data.id_31.astype(str).str.contains('android') & ~data.id_30.astype(str).str.contains('android') & ids.id_30.isin(['other','func','nan']),
        'id_30'
    ] = 'android'
    data.loc[
        data.id_31.astype(str).str.contains('mobile safari') & ~data.id_30.astype(str).str.contains('ios') & ids.id_30.isin(['other','func','nan']),
        'id_30'
    ] = 'ios'
    data.loc[
        data.id_31.astype(str).str.contains('ios') & ~data.id_30.astype(str).str.contains('ios') & ids.id_30.isin(['other','func','nan']),
        'id_30'
    ] = 'mac'
    data.loc[
        data.id_31.astype(str).str.contains('safari') & ~data.id_31.astype(str).str.contains('mobile safari') & ~data.id_30.astype(str).str.contains('mac') & ids.id_30.isin(['other','func','nan']),
        'id_30'
    ] = 'mac'
    data.loc[
        data.id_31.astype(str).str.contains('edge') & ~data.id_30.astype(str).str.contains('windows') & ids.id_30.isin(['other','func','nan']),
        'id_30'
    ] = 'windows'
    data.loc[
        data.id_31.astype(str).str.startswith('ie') & ~data.id_30.astype(str).str.contains('windows') & ids.id_30.isin(['other','func','nan']),
        'id_30'
    ] = 'windows'
    data.loc[
        data.id_31.astype(str).str.contains('windows') & ~data.id_30.astype(str).str.contains('windows') & ids.id_30.isin(['other','func','nan']),
        'id_30'
    ] = 'windows'
    
    # Special:
    data.loc[
        data.DeviceInfo.str.contains('windows') & ~data.id_30.astype(str).str.contains('android') & data.id_30.isin(['other','func','nan']),
        'id_30'
    ] = 'windows'
    data.loc[
        data.DeviceInfo.str.contains('android') & ~data.id_30.astype(str).str.contains('android') & data.id_30.isin(['other','func','nan']),
        'id_30'
    ] = 'android'
    
    data['manufacturer'] = np.nan
    data.loc[data.DeviceInfo.str.contains(r'^samsung|^sm-|^gt-|^sgh-|^sch-'), 'manufacturer'] = 'samsung'
    data.loc[data.DeviceInfo.str.contains(r'^lenovo'), 'manufacturer'] = 'lenovo'
    data.loc[data.DeviceInfo.str.contains(r'^ta-|nokia'), 'manufacturer'] = 'nokia'
    data.loc[data.DeviceInfo.str.contains(r'^lg|^lm-|^vs\d'), 'manufacturer'] = 'lg'
    data.loc[data.DeviceInfo.str.contains(r'^mot|^xt\d{4}'), 'manufacturer'] = 'motorolla'
    data.loc[data.DeviceInfo.str.contains(r'^android|nexus|pixel|oneplus'), 'manufacturer'] = 'google'
    data.loc[data.DeviceInfo.str.contains(r'htc'), 'manufacturer'] = 'htc'
    data.loc[data.DeviceInfo.str.contains(r'windows|microsoft|trident|rv:11.0|mddrjs'), 'manufacturer'] = 'microsoft'
    data.loc[data.DeviceInfo.str.contains(r'linux'), 'manufacturer'] = 'linux'
    data.loc[data.DeviceInfo.str.contains(r'ios device|macos|iphone'), 'manufacturer'] = 'apple'
    data.loc[data.DeviceInfo.str.contains(r'^[a-z]{3}-l|huawei|hi6210sft|^chc'), 'manufacturer'] = 'huawei'
    data.loc[data.DeviceInfo.str.contains(r'hisense'), 'manufacturer'] = 'hisense'
    data.loc[data.DeviceInfo.str.contains(r'redmi|^mi |^mi$'), 'manufacturer'] = 'xiaomi'
    data.loc[data.DeviceInfo.str.contains(r'ilium'), 'manufacturer'] = 'lanix'
    data.loc[data.DeviceInfo.str.contains(r'asus'), 'manufacturer'] = 'asus'
    data.loc[data.DeviceInfo.str.contains(r'zte|blade|^z\d{3} |^z\d{3}$'), 'manufacturer'] = 'zte'
    data.loc[data.DeviceInfo.str.contains(r'^kf'), 'manufacturer'] = 'amazon'
    data.loc[data.DeviceInfo.str.contains(r'^m4|m4tel'), 'manufacturer'] = 'm4tel'
    data.loc[data.DeviceInfo.str.contains(r'^\d{4}[a-z]$|^\d{4,}[a-z] |alcatel|one '), 'manufacturer'] = 'alcatel'
    data.loc[data.DeviceInfo.str.contains(r'^[a-z]\d{4}a$|^[a-z]\d{4}a |polaroid'), 'manufacturer'] = 'polaroid'
    data.loc[data.DeviceInfo.str.contains(r'^[a-z]\d{4}$|^[a-z]\d{4} |^sgp'), 'manufacturer'] = 'sony'
    data.loc[(data.DeviceInfo!='nan') & (data.manufacturer=='nan'), 'manufacturer'] = 'other'
    
    data.loc[
        data.id_30.isin(['other','func','nan']) & (data.manufacturer=='microsoft'),
        'id_30'
    ] = 'windows'
    
    data['platform'] = data.id_30.apply(lambda x: 'android' if 'android' in x else 'windows' if 'windows' in x else 'mac' if 'mac' in x else 'ios' if 'ios' in x else 'linux' if 'linux' in x else 'other') 
    data['platform_manufacturer'] = stringy(data.platform) + stringy(data.manufacturer)
    
    data['temp'] = data.id_33.astype(str).apply(lambda x: x.lower().split('x'))
    data['_rezx'] = data.temp.apply(lambda x: x[0] if len(x)==2 else np.nan).astype(np.float64)
    data['_rezy'] = data.temp.apply(lambda x: x[1] if len(x)==2 else np.nan).astype(np.float64)
    data['_aspect_ratio'] = data._rezx / data._rezy
    data['TransactionAmtCents'] = np.ceil(data.TransactionAmt) - np.floor(data.TransactionAmt)
    
    del data['temp']
    


    updateme('Email Features')
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

    updateme('Count + Label Encoding Everything')
    encodeit = [
        'CardID', 'DeviceID', 'PAccountID', 'RAccountID', 'ProductCD',
        'nulls_idn', 'nulls_trx',
        
        'P_emaildomain','R_emaildomain',
        'pemail_domain', 'pemail_ext', 'remail_domain', 'remail_ext', 'p_and_r_email',
        
        'platform','manufacturer','platform_manufacturer'
    ]
    for col in encodeit:
        mapper = {key:val for val,key in enumerate(data[col].unique())}
        if np.nan in mapper: mapper[np.nan] = np.nan # Keep Nans Nan
        data['ce_' + col] = data[col].map(data[col].value_counts(dropna=False))
        data['le_' + col] = data[col].map(mapper)
    data.drop(encodeit, axis=1, inplace=True)
    
    
    updateme('Time Features')    
    slope = 1 / (60*60*24) # sec/day
    for i in range(1,16):
        if i in [9]: continue
        feature = 'D' + str(i)
        data[feature+'_mfix'.format(i)] = np.round_(data[feature] - (data.TransactionDT - data.TransactionDT.min()) * slope)
        
    START_DATE     = '2017-12-01'
    startdate      = datetime.datetime.strptime(START_DATE, '%Y-%m-%d')
    data['tdt']    = data['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds = x)))
    data['tdow']   = data.tdt.dt.dayofweek
    data['thour']  = data.tdt.dt.hour
    data['tdate']  = data.tdt.dt.date
    data['flag_D9na']   = data.D9.isna()
    for col in ['tdow', 'thour']:
        mapper = {key:val for val,key in enumerate(data[col].unique())}
        if np.nan in mapper: mapper[np.nan] = np.nan # Keep Nans Nan
        data['ce_' + col] = data[col].map(data[col].value_counts(dropna=False))
        
    data['_OSRelease'] = (pd.to_datetime(data.tdt) - pd.to_datetime(data.id_30.map(id_30_dates))) / datetime.timedelta(days = 1)
      
    del data['tdt'], data['tdate'], data['D9']
    new_features = list(set(data.columns) - set(old_features))
    data.reset_index(drop=True, inplace=True)
    
    # NOTE: WE CAN TRY BUILDING ID_31 BROWSER RELEASE DATE
    return data, new_features


# # Start

# In[789]:


traintr = pd.read_csv('input/train_transaction.csv.zip')
trainid = pd.read_csv('input/train_identity.csv.zip')
testtr = pd.read_csv('input/test_transaction.csv.zip')
testid = pd.read_csv('input/test_identity.csv.zip')


# In[529]:





# In[536]:


# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# card1_of_addr1 = {}
# for sample_card1, sample_addr1 in tqdm(traintr[['card1','addr1']].values):
#     card1_of_addr1.setdefault(str(sample_card1), []).append(str(sample_addr1))

    
# addr1s = list(card1_of_addr1.keys())
# addr1s[:50]

# card1s_as_sentence = [' '.join(card1_of_addr1[addr1]) for addr1 in addr1s]
# card1s_as_matrix = CountVectorizer().fit_transform(card1s_as_sentence)
# topics_of_addr1s = LinearDiscriminantAnalysis(n_components=5).fit_transform(card1s_as_matrix)


# In[337]:


gc.collect()

trx_size = traintr.shape[0]
trans    = traintr.append(testtr, sort=False)
ids      = trainid.append(testid, sort=False)
trans.reset_index(drop=True, inplace=True)
ids.reset_index(drop=True, inplace=True)

data, new_features = build_cat_features(trans, ids)
gc.collect()


# In[338]:


data[new_features].dtypes.sort_values()


# In[339]:


data['noise0'] = np.random.normal(size=data.shape[0])
data['noise1'] = np.random.uniform(size=data.shape[0])

data_trn = data.iloc[:trx_size]
data_sub = data.iloc[trx_size:]


# In[340]:


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


# In[341]:


def experimemt(data, features, params, runs=3, train_frac=0.75, seed=123, expnum=0, retoof=False, max_trees=25):
    global trx_size
    
    # Run a quick experiment with 3 runs
    # Ideally with just a few features to evaluate their efficiency

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
    
    print(
        '\tCV roc score        : {0:.4f}, std: {1:.4f}.'.format(np.mean(roc_aucs), np.std(roc_aucs)),
        #'\n\tCV accuracy score   : {0:.4f}, std: {1:.4f}.'.format(np.mean(accuracies), np.std(accuracies)),
        #'\n\tCV recall score     : {0:.4f}, std: {1:.4f}.'.format(np.mean(recalls), np.std(recalls)),
        #'\n\tCV precision score  : {0:.4f}, std: {1:.4f}.'.format(np.mean(precisions), np.std(precisions)),
        #'\n\tCV f1 score         : {0:.4f}, std: {1:.4f}.'.format(np.mean(f1_scores), np.std(f1_scores))
    )

    if retoof:
        return oof, fi, np.mean(roc_aucs)
    
    return fi, np.mean(roc_aucs)


# In[429]:


from sklearn.ensemble import RandomForestClassifier

def test_cvs(data, feature):
    global trx_size
    
    data['covariate_shift'] = (np.arange(data.shape[0]) >= trx_size).astype(np.uint8)
    peek = data[~data[feature].isna()]
    
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


# # Run

# In[306]:


# First, let's find which V-vars have the BEST roc
# And select the top 1 as candidate (by vnullgroup) a
# continuous variables. and the top 1 that is not correlated to it
vrez = []
for vvar in ['V'+str(i) for i in range(1,340)]:
    print('\n',vvar)

    vrez.append(experimemt(
        traintr[[vvar, 'isFraud']],
        [vvar],
        params,
        runs=1,
        train_frac=0.75,
        seed=123,
        expnum=vvar
    )[0])
    
vrez = pd.concat(vrez, axis=0)


# In[308]:


# Select the best V* variable from each VNanGroup:
z = traintr[['V'+str(i) for i in range(1,340)]].isnull().sum().sort_values().reset_index().sort_values([0,'index'])
z.rename(columns={0:'cntna','index':'expnum'}, inplace=True)
vrez = vrez.merge(z[['cntna','expnum']], how='left', on='expnum')
maxroc = vrez.groupby('cntna').roc.max().reset_index()
maxroc.rename(columns={'roc':'maxroc'}, inplace=True)
vrez = vrez.merge(maxroc, how='left', on='cntna')
vrez['bestroc'] = False
vrez.loc[vrez.roc==vrez.maxroc,'bestroc'] = True
vrez_best = vrez[vrez.bestroc]

# Calculate each VNanGroup
vgroup = z.groupby('cntna').expnum.apply(list).reset_index()
for expnum, cntna in vrez_best[['expnum','cntna']].drop_duplicates().values:
    friends = vgroup[vgroup.cntna==cntna].expnum.iloc[0]

# Go through each BestFeature's VNanGroup + find the least correlated feature
enemies = []
for expnum, cntna in vrez_best[['expnum','cntna']].drop_duplicates(['expnum','cntna']).values:
    friends = vgroup[vgroup.cntna==cntna].expnum.iloc[0]
    friends = traintr[friends].corr()[expnum].abs()
    enemy = friends[friends==friends.min()].index[0]
    enemies.append(enemy)
enemies = pd.DataFrame({'expnum':vrez_best[['expnum']].drop_duplicates().values.flatten(), 'enemy':enemies})

vrez_best = vrez_best.merge(enemies, how='left', on='expnum')
vrez_conjugates = vrez[vrez.expnum.isin(vrez_best.enemy.unique())]

# vrez_best, vrez_conjugates
np.concatenate([vrez_best.expnum.unique(), vrez_best.enemy.unique()])


# In[347]:


# CATEGORICAL
# Trans
#     ProductCD
#     card1 - card6
#     addr1, addr2
#     P_emaildomain
#     R_emaildomain
#     M1 - M9
# ID
#     DeviceType
#     DeviceInfo
#     id_12 - id_38
#     id_31 = browser.
#     id_30 = operating system.
#     id_33 = screen resolution

test_cols = [c for c in traintr.columns if c not in ['TransactionID','isFraud','TransactionDT']]

cat_feats = [
    # Group With these
    'card1','card2','card3','card4','card5','card6',
    'addr1','addr2','M1','M2','M3','M4','M5','M6','M7','M8','M9'
] + [c for c in new_features if c[0]!='D' and c[0]!='_']

cont_feats = [
    'TransactionAmt', 'dist1',
    'D1','D2','D3','D4','D5','D6','D7',
    'D8','D10','D11','D12','D13','D14','D15',
] + [c for c in new_features if c[0]=='D' or c[0]=='_'] + vrez_best.expnum.unique().tolist() + vrez_best.enemy.unique().tolist()


# In[845]:


len(cont_feats), cont_feats


# In[349]:


def build_featureset(grp, data, grouper, compare):
    f1_std  = '{}_{}_std'.format(grouper, compare)
    f1_mean = '{}_{}_mean'.format(grouper, compare)
    f1_mean_diff = '{}_diff_{}'.format(compare, f1_mean)

    temp = grp[compare].agg(['mean','std']).reset_index()                  
    temp.rename(columns={'mean':f1_mean,'std':f1_std}, inplace=True)
    z = data[[grouper, compare, 'isFraud']].merge(temp, how='left', on=grouper)
    z[f1_mean_diff] = z[compare] - z[f1_mean]
    
    return z, [f1_mean, f1_mean_diff, f1_std]


# In[350]:


expnum = 0
rez = []
for grouper in cat_feats:
    print('—'*100)
    grp = data.groupby(grouper)
    
    for i,compare in enumerate(cont_feats):
        print('\n',grouper, i+1, 'of', len(cont_feats))
        z, feats = build_featureset(grp, data, grouper, compare)

        # Run experiment
        rez.append(experimemt(
            z,
            feats,
            params,
            runs=1,
            train_frac=0.75,
            seed=123,
            expnum='{}_{}'.format(grouper, compare)
        )[0])


# In[351]:


look = pd.concat(rez, axis=0)
look.reset_index(inplace=True, drop=True)
look['gps'] = look.gain / look.split
look = look[['feature','gain','split','gps','roc','expnum']].sort_values(['roc','expnum','gps','gain'], ascending=False)
look.to_csv('./feature_analysis.csv.gv', index=False)


# In[352]:


look.to_csv('./feature_analysis.csv.gv', index=False)


# # Exploration

# In[353]:


pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# In[361]:


def compare(data, a1,a2, b):
    a, afeats = build_featureset(data.groupby(a1), data, a1, a2)
    b, bfeats = build_featureset(data.groupby(b[0]), data, b[0], b[1])

    return pd.concat([
        a[['isFraud'] + afeats],
        b[bfeats]
    ], axis=1, sort=False).corr()


# In[362]:


compare(
    data,
    'ce_p_and_r_email', 'D3',
    ('ce_remail_domain', 'D3')
)


# In[364]:


plt.plot(look.roc.values)
plt.show()
look.head(1000)


# # Final Feature Building

# In[376]:


seed = 123

reg_params = {
    #'bagging_fraction': 0.80,
    'feature_fraction': 0.8,
    #'max_depth': 50,
    #'min_child_weight': 0.00298,
    #'min_data_in_leaf': 20,
    #'num_leaves': 382,
    #'reg_alpha': 1,
    #'reg_lambda': 2,
    
    #'max_depth': 20,
    #'num_leaves': 8,

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


# In[377]:


def combine(data, a1,a2, b):
    a, afeats = build_featureset(data.groupby(a1), data, a1, a2)
    
    for i, (b1, b2) in enumerate(b):
        b, bfeats = build_featureset(data.groupby(b1), data, b1, b2)
        
        if i==0:
            a = pd.concat([
                a[['isFraud'] + afeats],
                b[bfeats]
            ], axis=1, sort=False)
        else:
            a = pd.concat([a, b[bfeats]], axis=1, sort=False)
                
    return a


# In[435]:


z = combine(
    data,
    'ce_p_and_r_email',
    'D3',
    [
        ('M5', 'V258'),
        ('card2', 'TransactionAmt'),
        ('le_p_and_r_email', 'V283'),
        ('le_ProductCD', 'V102'),
        ('le_remail_domain', 'D2'),
        ('card1', 'D15_mfix'),
    ]
)


# In[436]:


y=experimemt(
    z,
    [f for f in z.columns if f!='isFraud'],
    reg_params,
    runs=3,
    train_frac=0.75,
    seed=123,
    expnum=1
)[0]


# In[ ]:


traintr.shape


# In[767]:


# a = data[data.index<trx_size]
# a=a[a.D1>0]

# b = data[data.index>=trx_size]
# b=b[b.D1>0]

# plt.hist(a.D1_mfix,100); plt.show()
# plt.hist(b.D1_mfix,100); plt.show()


# plt.figure(figsize=(16,9))
# plt.scatter(traintr[traintr.isFraud==0].TransactionDT, a[traintr.isFraud==0].D1_mfix, s=0.25, alpha=0.1)
# plt.scatter(traintr[traintr.isFraud==1].TransactionDT, a[traintr.isFraud==1].D1_mfix, s=0.25, alpha=1)
# plt.scatter(testtr.TransactionDT, b.D1_mfix, s=0.25, alpha=0.1)
# plt.show()

def graphit(feature):
    plt.figure(figsize=(16,9))
    plt.title(feature)
    plt.scatter(traintr[traintr.isFraud==0].TransactionDT, traintr[traintr.isFraud==0][feature], s=0.25, alpha=0.1)
    plt.scatter(traintr[traintr.isFraud==1].TransactionDT, traintr[traintr.isFraud==1][feature], s=0.25, alpha=1)
    plt.scatter(testtr.TransactionDT, testtr[feature], s=0.25, alpha=0.1)
#     plt.axhline(485, linewidth=1, linestyle='--', c='black')
#     plt.axhline(365, linewidth=1, linestyle='--', c='black')
#     plt.axhline(365*2, linewidth=1, linestyle='--', c='black')
    plt.show()

D1 = 'D1'
D2 = 'D5'
traintr[D1+'_-_'+D2] = traintr[D1]-traintr[D2]
testtr[D1+'_-_'+D2] = testtr[D1]-testtr[D2]
graphit(D1)
graphit(D2)
graphit(D1+'_-_'+D2)


# In[769]:


traintr[
    ~traintr[D1+'_-_'+D2].isna() & (traintr[D1+'_-_'+D2]!=0)
].isFraud.mean()


# In[656]:


# # (traintr.D12==traintr.D4).sum(), (~traintr.D12.isna()).sum()
traintr[
    ~traintr.D12.isna() &
    (traintr.D12!=traintr.D4)
].isFraud.sum() #/ traintr[~traintr.D12.isna() & (traintr.D12!=traintr.D4)].isFraud.mean()

# (traintr.D12==traintr.D4).sum() - (~traintr.D12.isna()).sum()


# In[658]:


publiclb = testtr[testtr.TransactionDT < testtr.TransactionDT.min() + 0.2*(testtr.TransactionDT.max()-testtr.TransactionDT.min())]
publiclb[~publiclb.D12.isna() & (publiclb.D12!=publiclb.D4)].shape[0] / publiclb.shape[0]


# In[660]:


testtr[~testtr.D12.isna() & (testtr.D12!=testtr.D4)].shape[0] #/ testtr.shape[0]


# In[623]:


(traintr[['D'+str(i) for i in range(1,16)]].isna().sum() / traintr.shape[0]).sort_values()


# In[626]:


test_cvs(data, 'D2')


# In[613]:


# D1,D2:   0.981311

# D3,D7:   0.818080, D3,D5: 0.707425
# D5,D7:   0.986496, D5,D3: 0.707425
# D7,D5:   0.986496, D7,D3: 0.818080

# D4,D12:  0.990999, D4,D6: 0.956966
# D6,D12:  0.976834, D6,D4: 0.956966
# D12,D4:  0.999999, D12,D6: 0.976834


# D8,D13:  0.521432
# D9,D8:   0.066085

# D10,D15: 0.712252
# D11,D15: 0.765000, NAN With D6,D7,D8,D9,D12,D13,D14

# D13,D8:  0.521432
# D14,D10: 0.336933
# D15,D11: 0.765000
cor = traintr[['D'+str(i) for i in range(1,16)]].corr()['D15'].sort_values()
cor


# In[554]:


slope = 1 / (60*60*24) # sec/day

z = traintr[['D1','isFraud','TransactionDT']].copy()
z['D1_mfix'] = np.round_(z.D1 - (z.TransactionDT - z.TransactionDT.min()) * slope)

START_DATE = '2017-12-01'
startdate  = datetime.datetime.strptime(START_DATE, '%Y-%m-%d')
z['tdt']   = z['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds = x)))
z['tdate'] = z.tdt.dt.date

min_D1mfix = z.groupby('tdate').D1_mfix.min().reset_index()
min_D1mfix.rename(columns={'D1_mfix':'md1mf'}, inplace=True)
z = z.merge(min_D1mfix,how='left', on='tdate')
z['newf'] = z.D1_mfix - z.md1mf


plt.figure(figsize=(16,9))
mask0 = z.isFraud==0
mask1 = z.isFraud==1
plt.scatter(z[mask0].TransactionDT, z[mask0].newf, s=0.25, alpha=0.1)
plt.scatter(z[mask1].TransactionDT, z[mask1].newf, s=0.25, alpha=0.75)
plt.show()


# In[479]:


#TODO: Map these to LE or CE... or binary
see = [    
#     'M4','M6', 'M2',
#     'D1', 'D1_mfix'
    'D3'
]

for col in see:
    print(col, test_cvs(data, col))


# In[389]:


gc.collect()

used_matches = [
    ('ce_p_and_r_email', 'D3'),
    ('card2', 'D1_mfix'), # TODO:  D1_mfix has CVS: 0.8579267280773009
    ('card1', 'V201'),
    ('M4', 'D15_mfix'),   # TODO: D15_mfix has CVS: 0.78573614980828
    ('M6', 'TransactionAmt'),
    ('card1', 'V283'),
    ('M2', 'D2_mfix'),    # TODO:  D2_mfix has CVS: 0.764469717377398
    ('card2', 'V258'),
]
best_roc = 0
best_features = None

tests = look[look.roc>.65].expnum.drop_duplicates().values
for i, test in enumerate(tests):
    print(i+1, '/', len(tests), '  —  ', test)
    pieces = test.split('_')
    cut = -1
    if pieces[-1] not in cont_feats:
        cut = -2
        
    a = '_'.join(pieces[:cut])
    b = '_'.join(pieces[cut:])
    if (a,b) in used_matches: continue
        
    # TODO: EZ Optimization - the value on the left shouldn't repeat, nor should the value on the right...
    # BUT MAYBE WE CANNOT DO THIS....
    # Also, we need to subtract std from final roc to stablize on features that score high but have low variance...
    z = combine(
        data,
        'ce_p_and_r_email', 'D3', # our best scoring feature we're trying to pair
        [
            ('card2', 'D1_mfix'),
            ('card1', 'V201'),
            ('M4', 'D15_mfix'),
            ('M6', 'TransactionAmt'),
            ('card1', 'V283'),
            ('M2', 'D2_mfix'),
            ('card2', 'V258'),
            (a,b)
        ]
    )
    
    _, roc = experimemt(
        z,
        [f for f in z.columns if f!='isFraud'],
        reg_params,
        runs=3,
        train_frac=0.75,
        seed=123,
        expnum=1,
        max_trees=75
    )
    
    if roc > best_roc:
        best_roc = roc
        best_features = (a,b)
        print('Found new best score!', best_roc, best_features, end='\n\n')


# In[ ]:


best_roc


# In[ ]:


best_features


# In[ ]:


- we have to redo these in order to ensure we're actually improving each time we add a feature pair
- try actually rotating the features, not just shearing it
- also try stdscaler or robustscaler transformations to get rid of covariate shift


# In[678]:


get_ipython().system('ls input')


# In[684]:


z = pd.read_csv('input/sample_submission.csv.zip')
z.loc[~testtr.D12.isna() & (testtr.D12!=testtr.D4), 'isFraud'] = 1 
z.to_csv('D4D12_Experiment.csv.zip', index=False)
z.head()


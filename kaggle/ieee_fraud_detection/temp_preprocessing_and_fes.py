# coding: utf-8

import gc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import time

from datetime import datetime, timedelta
from collections import deque, defaultdict
from itertools import islice
from tqdm import tqdm as tqdm
from contextlib import contextmanager

@contextmanager
def faith(title):
    start_time = time.time()
    yield
    print(">> {} - done in {:.0f}s".format(title, time.time() - start_time))

START_DATE = datetime.strptime('2017-12-01', '%Y-%m-%d') #or 30th Nov?
#START_DATE = datetime.strptime('2017-11-01', '%Y-%m-%d')

periods = ['7d', '14d']
min_instances = 1000
aggr_cols = [
   'addr1','card1','card2','card3','card4','card5','card6','ProductCD',
   'pemail_domain','remail_domain','pemail_ext', 'remail_ext',
]
country_map = {
'com':'us', 'net':'us', 'edu':'us', 'gmail':'us', 'mx': 'mx', 'es':'es', 'de':'de', 'fr':'fr','uk':'uk', 'jp':'jp'
}
domain = lambda x: x.split('.')[0]
pemail_country = lambda x: x.split('.')[-1]
USER_AGENTS = [
'Intel', 'Windows NT 6.1', 'Windows NT 6.2', 'Microsoft', 'Trident/7.0', 
'Touch', 'S.N.O.W.4', 'BOIE9', 'rv:11.0', 'rv:48.0', 'rv:52.0', 'rv:56.0',
'rv:57.0', 'rv:58.0', 'rv:59.0', 'rv:60.0', 'rv:61.0', 'rv:62.0', 'rv:63.0', 
'rv:64.0', 'rv:38.0', 'rv:51.0', 'rv:45.0', 'rv:42.0', 'rv:49.0', 'en-us',
'rv:41.0', 'rv:54.0', 'rv:47.0', 'rv:55.0', 'rv:31.0', 'rv:44.0', 'rv:53.0',
'rv:39.0', 'rv:35.0', 'rv:50.0', 'rv:37.0', 'rv:52.9', 'rv:46.0', 'rv:43.0',
'rv:29.0', 'rv:14.0', 'rv:33.0', 'rv:21.0', 'rv:27.0', 'rv:65.0', 'rv:28.0', 
'rv:60.1.0', 'es-us', 'es-es', 'es-mx', 'en-gb', 'Linux', 'MDDRJS',
'Android 5.1', 'Android 4.4.2', 'Android 6.0.1', 'Android 6.0', 'Android 7.0',
'Android', 'Android 8.0.0', 'Android 7.1.2', 'WOW64', 'ATT-IE11', 'MAMI', 'MALC',
'hp2015', 'Northwell', 'xs-Z47b7VqTMxs', 'QwestIE8', 'ATT', 'NetHelper70', 
'FunWebProducts', 'Lifesize', 'CPU'
]
CAT_FCOLS = ['card2', 'card3', 'card5', 'addr1', 'addr2', 'dist1', 'dist2']
C_FCOLS = [f'C{i}' for i in range(1, 15)]
D_FCOLS = [f'D{i}' for i in range(1, 16)]
V_FCOLS = [f'V{i}' for i in range(1, 340)] 
FLOAT64_TCOLS = CAT_FCOLS + C_FCOLS + D_FCOLS + V_FCOLS
FLOAT64_ICOLS = [f'id_0{i}' for i in range(1, 10)] + ['id_10', 'id_11', 'id_13', 'id_14', 'id_17', 'id_18', 'id_19', 'id_20', 'id_21', 'id_22', 'id_24', 'id_25', 'id_26', 'id_32']
id_30_dates = {
'Android 4.4.2':'2012-11-13','Android 5.0':'2014-11-12','Android 5.0.2':'2014-12-19',
'Android 5.1.1':'2015-04-21','Android 6.0':'2015-10-05','Android 6.0.1':'2015-12-07',
'Android 7.0':'2016-08-22','Android 7.1.1':'2016-12-05','Android 7.1.2':'2017-04-04',
'Android 8.0.0':'2017-08-21','Android 8.1.0':'2017-12-05','Android 9':'2018-08-06',
#Windows
'Windows XP':'2001-10-25','Windows Vista':'2006-11-08','Windows 7':'2009-10-22',
'Windows 8':'2012-10-26','Windows 8.1':'2013-10-17','Windows 10':'2015-07-29',
#MacOS
'Mac OS X 10.6': '2009-08-28','Mac OS X 10_6_8': '2011-06-23','Mac OS X 10_7_5': '2012-09-19',
'Mac OS X 10_8_5': '2013-09-12','Mac OS X 10.9': '2013-10-22','Mac OS X 10_9_5': '2014-09-17',
'Mac OS X 10.10': '2014-10-16','Mac OS X 10_10_5': '2015-08-13','Mac OS X 10.11': '2015-09-30',
'Mac OS X 10_11_3': '2016-01-19','Mac OS X 10_11_4': '2016-03-20','Mac OS X 10_11_5': '2016-05-16',
'Mac OS X 10_11_6': '2016-07-18','Mac OS X 10.12': '2016-09-20','Mac OS X 10_12': '2016-09-20',
'Mac OS X 10_12_1': '2016-10-24','Mac OS X 10_12_2': '2016-12-13','Mac OS X 10_12_3': '2017-01-23',
'Mac OS X 10_12_4': '2017-03-27','Mac OS X 10_12_5': '2017-05-15','Mac OS X 10_12_6': '2017-07-19',
'Mac OS X 10.13': '2017-09-25','Mac OS X 10_13_1': '2017-10-31','Mac OS X 10_13_2': '2017-12-06',
'Mac OS X 10_13_3': '2018-01-23','Mac OS X 10_13_4': '2018-03-29','Mac OS X 10_13_5': '2018-06-01',
'Mac OS X 10_13_6': '2018-07-09','Mac OS X 10.14': '2018-09-24','Mac OS X 10_14': '2018-09-24',
'Mac OS X 10_14_0': '2018-09-24','Mac OS X 10_14_1': '2018-10-30','Mac OS X 10_14_2': '2018-12-05',
#iOS
'iOS 9.3.5':'2016-08-25','iOS 10.0.2':'2016-09-23','iOS 10.1.1':'2016-10-31','iOS 10.2.0':'2016-12-12',
'iOS 10.2.1':'2017-01-23','iOS 10.3.1':'2017-04-03','iOS 10.3.2':'2017-05-15','iOS 10.3.3':'2017-07-19',
'iOS 11.0.0':'2017-08-19','iOS 11.0.1':'2017-08-26','iOS 11.0.2':'2017-10-03','iOS 11.0.3':'2017-10-11',
'iOS 11.1.0':'2017-10-31','iOS 11.1.1':'2017-11-08','iOS 11.1.2':'2017-11-16','iOS 11.2.0':'2017-12-02',
'iOS 11.2.1':'2017-12-13','iOS 11.2.2':'2018-01-08','iOS 11.2.5':'2018-01-23','iOS 11.2.6':'2018-02-19',
'iOS 11.3.0':'2018-03-29','iOS 11.3.1':'2018-04-24','iOS 11.4.0':'2018-05-29','iOS 11.4.1':'2018-07-09',
'iOS 12.0.0':'2018-08-17','iOS 12.0.1':'2018-09-08','iOS 12.1.0':'2018-09-30','iOS 12.1.1':'2018-12-05',
'iOS 12.1.2':'2018-12-20',
}
id_30_dates = {k.lower():v for k,v in id_30_dates.items()}

with faith('1. Loading Data Hold On ....') as f:

	df_train_identity = pd.read_csv('../input/train_identity.csv', dtype=dict.fromkeys(FLOAT64_ICOLS, np.float32),)
	df_test_identity = pd.read_csv('../input/test_identity.csv', dtype=dict.fromkeys(FLOAT64_ICOLS, np.float32),)
	df_train_transaction = pd.read_csv('../input/train_transaction.csv', dtype=dict.fromkeys(FLOAT64_TCOLS, np.float32),)
	df_test_transaction = pd.read_csv('../input/test_transaction.csv', dtype=dict.fromkeys(FLOAT64_TCOLS, np.float32),)
	X_train = pd.merge(df_train_transaction, df_train_identity, how='left', on='TransactionID')
	X_test = pd.merge(df_test_transaction, df_test_identity, how='left', on='TransactionID')
	org_cols = X_train.columns.tolist()

	print('Number of Training Examples = {}'.format(df_train_transaction.shape[0]))
	print('Number of Test Examples = {}\\n'.format(df_test_transaction.shape[0]))
	print('Number of Training Examples with Identity = {}'.format(df_train_identity.shape[0]))
	print('Number of Test Examples with Identity = {}\\n'.format(df_test_identity.shape[0]))
	print('Training X Shape = {}'.format(X_train.shape))
	print('Training y Shape = {}'.format(X_train['isFraud'].shape))
	print('Test X Shape = {}\\n'.format(X_test.shape))
	print('Training Set Memory Usage = {:.2f} MB'.format(X_train.memory_usage().sum() / 1024**2))
	print('Test Set Memory Usage = {:.2f} MB\\n'.format(X_test.memory_usage().sum() / 1024**2))
	del df_train_identity, df_test_identity, df_train_transaction, df_test_transaction

with faith('2. Adding simple time feats like minute hour etc will be dropped later for sure') as f:
	for df in tqdm([X_train, X_test]):
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

with faith('3. Fixing id_30 and DeviceInfo and inferring more vals for other cols etc...') as f:
	for df in tqdm([X_train, X_test]):

	    ########## DeviceInfo ##########    
	    
	    # Finding DeviceInfo from id_31
	    df.loc[df.query('DeviceInfo.isnull() and id_31.str.contains("mobile safari")', engine='python').index, 'DeviceInfo'] = 'iOS Device'
	    df.loc[df.query('DeviceInfo.isnull() and id_31.str.contains("for ios")', engine='python').index, 'DeviceInfo'] = 'iOS Device'
	    df.loc[df.query('DeviceInfo.isnull() and id_31.str.startswith("google search application")', engine='python').index, 'DeviceInfo'] = 'iOS Device'
	    df.loc[df.query('DeviceInfo.isnull() and id_31 == "google"', engine='python').index, 'DeviceInfo'] = 'iOS Device'
	    df.loc[df.query('DeviceInfo.isnull() and id_31 == "safari"', engine='python').index, 'DeviceInfo'] = 'MacOS'
	    df.loc[df.query('DeviceInfo.isnull() and id_31 == "safari generic"', engine='python').index, 'DeviceInfo'] = 'MacOS'
	    df.loc[df.query('DeviceInfo.isnull() and id_31 == "safari 9.0"', engine='python').index, 'DeviceInfo'] = 'MacOS'
	    df.loc[df.query('DeviceInfo.isnull() and id_31 == "safari 10.0"', engine='python').index, 'DeviceInfo'] = 'MacOS'
	    df.loc[df.query('DeviceInfo.isnull() and id_31 == "safari 11.0"', engine='python').index, 'DeviceInfo'] = 'MacOS'
	    df.loc[df.query('DeviceInfo.isnull() and id_31 == "safari 12.0"', engine='python').index, 'DeviceInfo'] = 'MacOS'    
	    
	    ########## DeviceType ##########    
	    
	    # Finding DeviceType from DeviceInfo 
	    df.loc[df.query('DeviceType.isnull() and id_31 == "ie 11.0 for desktop"', engine='python').index, 'DeviceType'] = 'desktop'
	    df.loc[df.query('DeviceType.isnull() and id_31 == "chrome 65.0"', engine='python').index, 'DeviceType'] = 'desktop'
	    df.loc[df.query('DeviceType.isnull() and id_31 == "ie 11.0 for tablet"', engine='python').index, 'DeviceType'] = 'desktop'
	    # Finding DeviceType from id_31
	    df.loc[df.query('DeviceType.isnull() and ~DeviceInfo.isnull()', engine='python').index, 'DeviceType'] = 'desktop'
	   
	    ########## id_30 ##########    
	    
	    # Finding id_30 from DeviceInfo parsing errors
	    df.loc[df.query('id_30.isnull() and DeviceInfo == "Linux x86_64"', engine='python').index, 'id_30'] = 'Linux'
	    df.loc[df.query('id_30.isnull() and DeviceInfo == "Linux i686"', engine='python').index, 'id_30'] = 'Linux'    
	    df.loc[df.query('id_30.isnull() and DeviceInfo == "BOIE9"', engine='python').index, 'id_30'] = 'Windows 7'
	    df.loc[df.query('id_30.isnull() and DeviceInfo == "MDDRJS"', engine='python').index, 'id_30'] = 'Windows 7'
	    df.loc[df.query('id_30.isnull() and DeviceInfo == "Windows NT 6.1"', engine='python').index, 'id_30'] = 'Windows 7'
	    df.loc[df.query('id_30.isnull() and DeviceInfo == "Windows NT 6.2"', engine='python').index, 'id_30'] = 'Windows 8'
	    df.loc[df.query('id_30.isnull() and DeviceInfo == "Microsoft"', engine='python').index, 'id_30'] = 'Windows 10'
	    df.loc[df.query('id_30.isnull() and DeviceInfo == "Windows" and id_31.str.startswith("edge")', engine='python').index, 'id_30'] = 'Windows 10'
	    df.loc[df.query('id_30.isnull() and DeviceInfo == "Android 5.1"', engine='python').index, 'id_30'] = 'Android 5.1'
	    df.loc[df.query('id_30.isnull() and DeviceInfo == "Android 4.4.2"', engine='python').index, 'id_30'] = 'Android 4.4.2'
	    df.loc[df.query('id_30.isnull() and DeviceInfo == "Android 5.1.1"', engine='python').index, 'id_30'] = 'Android 5.1.1'
	    df.loc[df.query('id_30.isnull() and DeviceInfo == "Android 6.0.1"', engine='python').index, 'id_30'] = 'Android 6.0.1'
	    df.loc[df.query('id_30.isnull() and DeviceInfo == "Android 6.0"', engine='python').index, 'id_30'] = 'Android 6.0'
	    df.loc[df.query('id_30.isnull() and DeviceInfo == "Android 7.0"', engine='python').index, 'id_30'] = 'Android 7.0'
	    df.loc[df.query('id_30.isnull() and DeviceInfo == "Android"', engine='python').index, 'id_30'] = 'Android'
	    df.loc[df.query('id_30.isnull() and DeviceInfo == "Android 7.1.2"', engine='python').index, 'id_30'] = 'Android 7.1.2'
	    df.loc[df.query('id_30.isnull() and DeviceInfo == "Android 8.0.0"', engine='python').index, 'id_30'] = 'Android 8.0.0'
	    
	    # Finding id_30 from id_31 parsing errors
	    df.loc[df.query('id_30.isnull() and id_31 == "Generic/Android 7.0"', engine='python').index, 'id_30'] = 'Android 7.0'
	    df.loc[df.query('id_30.isnull() and id_31.str.startswith("edge")', engine='python').index, 'id_30'] = 'Windows 10'
	    
	    # Finding id_30 from Android Build Numbers
	    # Android devices without Build Numbers are labeled as Android
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("Build/Huawei")', engine='python').index, 'id_30'] = 'Android'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("Build/HUAWEI")', engine='python').index, 'id_30'] = 'Android'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("Build/S100")', engine='python').index, 'id_30'] = 'Android'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("Build/Vision")', engine='python').index, 'id_30'] = 'Android'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("Build/HONOR")', engine='python').index, 'id_30'] = 'Android'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("Build/Lenovo")', engine='python').index, 'id_30'] = 'Android'
	    
	    # Android devices with Build Numbers are mapped with their correct id_30 values
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("IML74K")', engine='python').index, 'id_30'] = 'Android 4.0'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("JZO54K")', engine='python').index, 'id_30'] = 'Android 4.1'    
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("6.2.A.1.100")', engine='python').index, 'id_30'] = 'Android 4.1'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("9.8.2I-50_SML-25")', engine='python').index, 'id_30'] = 'Android 4.1'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("JDQ39")', engine='python').index, 'id_30'] = 'Android 4.2'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("JLS36C")', engine='python').index, 'id_30'] = 'Android 4.3'    
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("KTU84M")', engine='python').index, 'id_30'] = 'Android 4.4'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("KTU84P")', engine='python').index, 'id_30'] = 'Android 4.4'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("KOT49H")', engine='python').index, 'id_30'] = 'Android 4.4'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("KOT49I")', engine='python').index, 'id_30'] = 'Android 4.4'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("KVT49L")', engine='python').index, 'id_30'] = 'Android 4.4'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("KXB21.14-L1.40")', engine='python').index, 'id_30'] = 'Android 4.4'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("KXB20.9-1.10-1.24-1.1")', engine='python').index, 'id_30'] = 'Android 4.4'    
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("KXC21.5-40")', engine='python').index, 'id_30'] = 'Android 4.4' 
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("24.0.A.5.14")', engine='python').index, 'id_30'] = 'Android 4.4'  
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("SU6-7.7")', engine='python').index, 'id_30'] = 'Android 4.4'    
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("26.1.A.3.111")', engine='python').index, 'id_30'] = 'Android 5.0'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("27.1.A.1.81")', engine='python').index, 'id_30'] = 'Android 5.0'    
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("LRX21R")', engine='python').index, 'id_30'] = 'Android 5.0'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("LRX22C")', engine='python').index, 'id_30'] = 'Android 5.0'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("LRX21V")', engine='python').index, 'id_30'] = 'Android 5.0'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("LRX21M")', engine='python').index, 'id_30'] = 'Android 5.0'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("LRX21Y")', engine='python').index, 'id_30'] = 'Android 5.0'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("LRX21T")', engine='python').index, 'id_30'] = 'Android 5.0'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("LRX22G")', engine='python').index, 'id_30'] = 'Android 5.0'    
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("LPBS23.13-57-2")', engine='python').index, 'id_30'] = 'Android 5.1'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("LPCS23.13-56-5")', engine='python').index, 'id_30'] = 'Android 5.1'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("LPBS23.13-56-2")', engine='python').index, 'id_30'] = 'Android 5.1'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("LPAS23.12-21.7-1")', engine='python').index, 'id_30'] = 'Android 5.1'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("LPCS23.13-34.8-3")', engine='python').index, 'id_30'] = 'Android 5.1'  
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("E050L")', engine='python').index, 'id_30'] = 'Android 5.1'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("L050U")', engine='python').index, 'id_30'] = 'Android 5.1'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("LMY48B")', engine='python').index, 'id_30'] = 'Android 5.1'    
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("LMY47D")', engine='python').index, 'id_30'] = 'Android 5.1'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("LMY47I")', engine='python').index, 'id_30'] = 'Android 5.1'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("LMY47V")', engine='python').index, 'id_30'] = 'Android 5.1'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("LVY48F")', engine='python').index, 'id_30'] = 'Android 5.1'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("LMY47O")', engine='python').index, 'id_30'] = 'Android 5.1'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("LMY47X")', engine='python').index, 'id_30'] = 'Android 5.1'   
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("10.7.A.0.222")', engine='python').index, 'id_30'] = 'Android 5.1'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("14.6.A.0.368")', engine='python').index, 'id_30'] = 'Android 5.1'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("14.6.A.1.236")', engine='python').index, 'id_30'] = 'Android 5.1'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("18.6.A.0.182")', engine='python').index, 'id_30'] = 'Android 5.1'    
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("19.4.A.0.182")', engine='python').index, 'id_30'] = 'Android 5.1'  
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("29.1.A.0.101")', engine='python').index, 'id_30'] = 'Android 5.1'    
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("MPIS24.241-15.3-7")', engine='python').index, 'id_30'] = 'Android 6.0'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("MPDS24.65-33-1-30")', engine='python').index, 'id_30'] = 'Android 6.0'    
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("26.3.A.1.33")', engine='python').index, 'id_30'] = 'Android 6.0'    
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("27.3.A.0.165")', engine='python').index, 'id_30'] = 'Android 6.0'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("27.3.A.0.129")', engine='python').index, 'id_30'] = 'Android 6.0'    
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("27.3.A.0.173")', engine='python').index, 'id_30'] = 'Android 6.0'    
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("29.2.A.0.166")', engine='python').index, 'id_30'] = 'Android 6.0'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("36.0.A.2.146")', engine='python').index, 'id_30'] = 'Android 6.0'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("33.2.A.3.81")', engine='python').index, 'id_30'] = 'Android 6.0' 
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("33.2.A.4.70")', engine='python').index, 'id_30'] = 'Android 6.0'    
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("23.5.A.1.291")', engine='python').index, 'id_30'] = 'Android 6.0'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("37.0.A.2.108")', engine='python').index, 'id_30'] = 'Android 6.0'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("37.0.A.2.248")', engine='python').index, 'id_30'] = 'Android 6.0'    
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("30.2.A.1.21")', engine='python').index, 'id_30'] = 'Android 6.0' 
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("35.0.D.2.25")', engine='python').index, 'id_30'] = 'Android 6.0'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("M4B30Z")', engine='python').index, 'id_30'] = 'Android 6.0'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("MPB24.65-34-3")', engine='python').index, 'id_30'] = 'Android 6.0' 
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("MPI24.65-25")', engine='python').index, 'id_30'] = 'Android 6.0'     
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("MPDS24.107-52-11")', engine='python').index, 'id_30'] = 'Android 6.0' 
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("MPIS24.65-33.1-2-10")', engine='python').index, 'id_30'] = 'Android 6.0' 
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("MPIS24.65-33.1-2-16")', engine='python').index, 'id_30'] = 'Android 6.0' 
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("MPIS24.65-25.1-19")', engine='python').index, 'id_30'] = 'Android 6.0' 
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("MPIS24.107-55-2-17")', engine='python').index, 'id_30'] = 'Android 6.0'     
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("MPIS24.241-2.35-1-17")', engine='python').index, 'id_30'] = 'Android 6.0'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("MPIS24.241-15.3-26")', engine='python').index, 'id_30'] = 'Android 6.0'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("MPI24.65-33.1-2")', engine='python').index, 'id_30'] = 'Android 6.0'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("MCG24.251-5-5")', engine='python').index, 'id_30'] = 'Android 6.0'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("MPB24.65-34")', engine='python').index, 'id_30'] = 'Android 6.0'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("MPD24.107-52")', engine='python').index, 'id_30'] = 'Android 6.0'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("MPD24.65-25")', engine='python').index, 'id_30'] = 'Android 6.0'    
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("MPDS24.107-52-5")', engine='python').index, 'id_30'] = 'Android 6.0'   
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("MPDS24.65-33-1-3")', engine='python').index, 'id_30'] = 'Android 6.0'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("IEXCNFN5902303111S")', engine='python').index, 'id_30'] = 'Android 6.0'    
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("MPD24.65-33")', engine='python').index, 'id_30'] = 'Android 6.0'     
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("MHC19Q")', engine='python').index, 'id_30'] = 'Android 6.0'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("MMB28B")', engine='python').index, 'id_30'] = 'Android 6.0'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("MOB30M")', engine='python').index, 'id_30'] = 'Android 6.0'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("MMB29K")', engine='python').index, 'id_30'] = 'Android 6.0'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("MRA58K")', engine='python').index, 'id_30'] = 'Android 6.0'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("MMB29M")', engine='python').index, 'id_30'] = 'Android 6.0'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("MMB29T")', engine='python').index, 'id_30'] = 'Android 6.0'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("MXB48T")', engine='python').index, 'id_30'] = 'Android 6.0'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("NRD90M")', engine='python').index, 'id_30'] = 'Android 7.0'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("NRD90N")', engine='python').index, 'id_30'] = 'Android 7.0'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("NRD90U")', engine='python').index, 'id_30'] = 'Android 7.0'    
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("33.3.A.1.97")', engine='python').index, 'id_30'] = 'Android 7.0'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("33.3.A.1.115")', engine='python').index, 'id_30'] = 'Android 7.0' 
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("34.2.A.2.47")', engine='python').index, 'id_30'] = 'Android 7.0'     
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("36.1.A.1.86")', engine='python').index, 'id_30'] = 'Android 7.0'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("40.0.A.6.175")', engine='python').index, 'id_30'] = 'Android 7.0'    
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("40.0.A.6.135")', engine='python').index, 'id_30'] = 'Android 7.0'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("40.0.A.6.189")', engine='python').index, 'id_30'] = 'Android 7.0'    
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("42.0.A.4.101")', engine='python').index, 'id_30'] = 'Android 7.0'  
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("42.0.A.4.167")', engine='python').index, 'id_30'] = 'Android 7.0'     
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("43.0.A.5.79")', engine='python').index, 'id_30'] = 'Android 7.0'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("43.0.A.7.25")', engine='python').index, 'id_30'] = 'Android 7.0'    
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("43.0.A.7.70")', engine='python').index, 'id_30'] = 'Android 7.0'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("43.0.A.7.55")', engine='python').index, 'id_30'] = 'Android 7.0'     
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("HONORBLN-L24")', engine='python').index, 'id_30'] = 'Android 7.0'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("HONORDLI-L22")', engine='python').index, 'id_30'] = 'Android 7.0'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("NPHS25.200-15-8")', engine='python').index, 'id_30'] = 'Android 7.0'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("NPHS25.200-23-1")', engine='python').index, 'id_30'] = 'Android 7.0'    
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("NPNS25.137-92-4")', engine='python').index, 'id_30'] = 'Android 7.0'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("NPNS25.137-92-8")', engine='python').index, 'id_30'] = 'Android 7.0'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("NPNS25.137-15-11")', engine='python').index, 'id_30'] = 'Android 7.0'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("NPNS25.137-93-14")', engine='python').index, 'id_30'] = 'Android 7.0'    
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("NPP25.137-82")', engine='python').index, 'id_30'] = 'Android 7.0'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("NPN25.137-72")', engine='python').index, 'id_30'] = 'Android 7.0'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("NPPS25.137-15-11")', engine='python').index, 'id_30'] = 'Android 7.0'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("NPP25.137-33")', engine='python').index, 'id_30'] = 'Android 7.0'    
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("NPP25.137-72")', engine='python').index, 'id_30'] = 'Android 7.0'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("NPJ25.93-14.5")', engine='python').index, 'id_30'] = 'Android 7.0' 
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("NPJS25.93-14-10")', engine='python').index, 'id_30'] = 'Android 7.0' 
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("NPJ25.93-14")', engine='python').index, 'id_30'] = 'Android 7.0'  
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("NPJS25.93-14-15")', engine='python').index, 'id_30'] = 'Android 7.0'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("NPP25.137-93")', engine='python').index, 'id_30'] = 'Android 7.0'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("NPPS25.137-93-8")', engine='python').index, 'id_30'] = 'Android 7.0'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("NPKS25.200-17-8")', engine='python').index, 'id_30'] = 'Android 7.0'  
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("NPKS25.200-12-9")', engine='python').index, 'id_30'] = 'Android 7.0'    
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("NPP25.137-15")', engine='python').index, 'id_30'] = 'Android 7.0'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("NPP25.137-38")', engine='python').index, 'id_30'] = 'Android 7.0'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("NPJ25.93-14.7")', engine='python').index, 'id_30'] = 'Android 7.0'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("NPJS25.93-14-13")', engine='python').index, 'id_30'] = 'Android 7.0'     
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("NPN25.137-35")', engine='python').index, 'id_30'] = 'Android 7.0'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("NPN25.137-15")', engine='python').index, 'id_30'] = 'Android 7.0'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("NPN25.137-92")', engine='python').index, 'id_30'] = 'Android 7.0'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("NPN25.137-82")', engine='python').index, 'id_30'] = 'Android 7.0'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("NCK25.118-10.5")', engine='python').index, 'id_30'] = 'Android 7.0'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("NPPS25.137-93-4")', engine='python').index, 'id_30'] = 'Android 7.0' 
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("NPPS25.137-93-12")', engine='python').index, 'id_30'] = 'Android 7.0'     
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("NPPS25.137-93-14")', engine='python').index, 'id_30'] = 'Android 7.0'    
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("NPJS25.93-14-18")', engine='python').index, 'id_30'] = 'Android 7.0'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("NPJS25.93-14-8.1-4")', engine='python').index, 'id_30'] = 'Android 7.0'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("NPJS25.93-14-8")', engine='python').index, 'id_30'] = 'Android 7.0'  
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("NPJS25.93-14.7-8")', engine='python').index, 'id_30'] = 'Android 7.0'    
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("NPNS25.137-92-10")', engine='python').index, 'id_30'] = 'Android 7.0'  
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("NPNS25.137-92-14")', engine='python').index, 'id_30'] = 'Android 7.0'    
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("NJH47F")', engine='python').index, 'id_30'] = 'Android 7.1' 
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("N6F27M")', engine='python').index, 'id_30'] = 'Android 7.1' 
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("NMF26O")', engine='python').index, 'id_30'] = 'Android 7.1' 
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("NMF26V")', engine='python').index, 'id_30'] = 'Android 7.1' 
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("NMF26F")', engine='python').index, 'id_30'] = 'Android 7.1' 
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("N2G47H")', engine='python').index, 'id_30'] = 'Android 7.1' 
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("NMF26X")', engine='python').index, 'id_30'] = 'Android 7.1'  
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("NMF26Q")', engine='python').index, 'id_30'] = 'Android 7.1'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("N2G48C")', engine='python').index, 'id_30'] = 'Android 7.1'    
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("32.4.A.1.54")', engine='python').index, 'id_30'] = 'Android 7.1'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("34.3.A.0.252")', engine='python').index, 'id_30'] = 'Android 7.1'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("34.3.A.0.228")', engine='python').index, 'id_30'] = 'Android 7.1'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("34.3.A.0.238")', engine='python').index, 'id_30'] = 'Android 7.1'    
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("41.2.A.7.76")', engine='python').index, 'id_30'] = 'Android 7.1'    
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("NDNS26.118-23-12-3")', engine='python').index, 'id_30'] = 'Android 7.1'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("NCQ26.69-56")', engine='python').index, 'id_30'] = 'Android 7.1'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("NMA26.42-19")', engine='python').index, 'id_30'] = 'Android 7.1'   
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("NPN26.118-22-2")', engine='python').index, 'id_30'] = 'Android 7.1'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("NPD26.48-24-1")', engine='python').index, 'id_30'] = 'Android 7.1'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("NPSS26.118-19-14")', engine='python').index, 'id_30'] = 'Android 7.1'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("NPS26.118-19")', engine='python').index, 'id_30'] = 'Android 7.1'     
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("NMA26.42-167")', engine='python').index, 'id_30'] = 'Android 7.1'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("NMA26.42-152")', engine='python').index, 'id_30'] = 'Android 7.1'    
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("NMA26.42-142")', engine='python').index, 'id_30'] = 'Android 7.1'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("NMA26.42-69")', engine='python').index, 'id_30'] = 'Android 7.1'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("NMA26.42-11-3")', engine='python').index, 'id_30'] = 'Android 7.1'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("NMA26.42-157")', engine='python').index, 'id_30'] = 'Android 7.1'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("NMA26.42-162")', engine='python').index, 'id_30'] = 'Android 7.1'    
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("NPW26.83-42")', engine='python').index, 'id_30'] = 'Android 7.1'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("NPW26.83-18-2-0-4")', engine='python').index, 'id_30'] = 'Android 7.1'    
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("NDQS26.69-64-2")', engine='python').index, 'id_30'] = 'Android 7.1' 
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("NDQS26.69-23-2-3")', engine='python').index, 'id_30'] = 'Android 7.1'    
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("NPIS26.48-36-2")', engine='python').index, 'id_30'] = 'Android 7.1'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("NPIS26.48-43-2")', engine='python').index, 'id_30'] = 'Android 7.1'    
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("NPSS26.118-19-22")', engine='python').index, 'id_30'] = 'Android 7.1'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("NPI26.48-36")', engine='python').index, 'id_30'] = 'Android 7.1'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("NPIS26.48-36-5")', engine='python').index, 'id_30'] = 'Android 7.1'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("NPIS26.48-38-3")', engine='python').index, 'id_30'] = 'Android 7.1'    
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("NPLS26.118-20-5-3")', engine='python').index, 'id_30'] = 'Android 7.1'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("NPSS26.118-19-6")', engine='python').index, 'id_30'] = 'Android 7.1'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("NPS26.74-16-1")', engine='python').index, 'id_30'] = 'Android 7.1'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("NPSS26.118-19-11")', engine='python').index, 'id_30'] = 'Android 7.1'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("NPSS26.118-19-4")', engine='python').index, 'id_30'] = 'Android 7.1'    
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("NDSS26.118-23-19-6")', engine='python').index, 'id_30'] = 'Android 7.1'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("NDQ26.69-64-9")', engine='python').index, 'id_30'] = 'Android 7.1'    
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("NPLS26.118-20-5-11")', engine='python').index, 'id_30'] = 'Android 7.1'    
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("34.4.A.2.19")', engine='python').index, 'id_30'] = 'Android 8.0'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("34.4.A.2.107")', engine='python').index, 'id_30'] = 'Android 8.0'    
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("47.1.A.12.270")', engine='python').index, 'id_30'] = 'Android 8.0'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("47.1.A.5.51")', engine='python').index, 'id_30'] = 'Android 8.0'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("48.1.A.2.21")', engine='python').index, 'id_30'] = 'Android 8.0'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("48.1.A.2.50")', engine='python').index, 'id_30'] = 'Android 8.0' 
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("50.1.A.10.40")', engine='python').index, 'id_30'] = 'Android 8.0'     
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("R16NW")', engine='python').index, 'id_30'] = 'Android 8.0'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("OCLS27.76-69-6")', engine='python').index, 'id_30'] = 'Android 8.0'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("OPR1.170623.032")', engine='python').index, 'id_30'] = 'Android 8.0'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("OPR5.170623.014")', engine='python').index, 'id_30'] = 'Android 8.0'    
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("OPR6.170623.013")', engine='python').index, 'id_30'] = 'Android 8.0'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("OPP27.61-14-4")', engine='python').index, 'id_30'] = 'Android 8.0'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("OPP27.91-25")', engine='python').index, 'id_30'] = 'Android 8.0'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("OPP27.91-87")', engine='python').index, 'id_30'] = 'Android 8.0'    
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("OPP27.91-140")', engine='python').index, 'id_30'] = 'Android 8.0'   
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("OPS27.82-41")', engine='python').index, 'id_30'] = 'Android 8.0'     
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("OPSS27.76-12-25-7")', engine='python').index, 'id_30'] = 'Android 8.0'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("OPS27.76-12-25")', engine='python').index, 'id_30'] = 'Android 8.0' 
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("OPSS27.82-87-3")', engine='python').index, 'id_30'] = 'Android 8.0' 
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("OPS27.82-87")', engine='python').index, 'id_30'] = 'Android 8.0' 
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("OPS27.82-72")', engine='python').index, 'id_30'] = 'Android 8.0' 
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("OPSS27.76-12-25-3")', engine='python').index, 'id_30'] = 'Android 8.0'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("OPP27.91-143")', engine='python').index, 'id_30'] = 'Android 8.0'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("OPR1.170623.027")', engine='python').index, 'id_30'] = 'Android 8.0'    
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("OPNS27.76-12-22-9")', engine='python').index, 'id_30'] = 'Android 8.0'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("OPNS27.76-12-22-3")', engine='python').index, 'id_30'] = 'Android 8.0'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("OPN27.76-12-22")', engine='python').index, 'id_30'] = 'Android 8.0'    
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("O00623")', engine='python').index, 'id_30'] = 'Android 8.0'    
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("OPW27.57-40")', engine='python').index, 'id_30'] = 'Android 8.0'     
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("OPW27.113-89")', engine='python').index, 'id_30'] = 'Android 8.0'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("OPWS27.113-25-4")', engine='python').index, 'id_30'] = 'Android 8.0'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("OPWS27.57-40-14")', engine='python').index, 'id_30'] = 'Android 8.0'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("OPWS27.57-40-17")', engine='python').index, 'id_30'] = 'Android 8.0'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("OPWS27.113-89-2")', engine='python').index, 'id_30'] = 'Android 8.0'    
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("OPP27.91-72")', engine='python').index, 'id_30'] = 'Android 8.0'    
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("OPP27.91-122")', engine='python').index, 'id_30'] = 'Android 8.0'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("OPP27.91-146")', engine='python').index, 'id_30'] = 'Android 8.0'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("OPWS27.57-40-6")', engine='python').index, 'id_30'] = 'Android 8.0'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("OPWS27.57-40-22")', engine='python').index, 'id_30'] = 'Android 8.0'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("OPR1.170623.026")', engine='python').index, 'id_30'] = 'Android 8.0'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("OPR4.170623.006")', engine='python').index, 'id_30'] = 'Android 8.0'    
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("OPM1.171019.012")', engine='python').index, 'id_30'] = 'Android 8.1'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("OPM3.171019.013")', engine='python').index, 'id_30'] = 'Android 8.1'    
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("OPM1.171019.011")', engine='python').index, 'id_30'] = 'Android 8.1'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("OPM1.171019.021")', engine='python').index, 'id_30'] = 'Android 8.1'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("OPM1.171019.019")', engine='python').index, 'id_30'] = 'Android 8.1'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("OPM1.171019.026")', engine='python').index, 'id_30'] = 'Android 8.1'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("OPM7.181005.003")', engine='python').index, 'id_30'] = 'Android 8.1'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("OPM2.171026.006.C1")', engine='python').index, 'id_30'] = 'Android 8.1'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("OPM4.171019.021.P1")', engine='python').index, 'id_30'] = 'Android 8.1'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("OPM2.171026.006.H1")', engine='python').index, 'id_30'] = 'Android 8.1'    
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("OPM2.171026.006.G1")', engine='python').index, 'id_30'] = 'Android 8.1'    
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("OPPS28.85-13-2")', engine='python').index, 'id_30'] = 'Android 8.1'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("OPW28.70-22")', engine='python').index, 'id_30'] = 'Android 8.1'    
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("OPS28.85-13")', engine='python').index, 'id_30'] = 'Android 8.1'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("OPSS28.85-13-3")', engine='python').index, 'id_30'] = 'Android 8.1'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("O11019")', engine='python').index, 'id_30'] = 'Android 8.1'    
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("OPGS28.54-19-2")', engine='python').index, 'id_30'] = 'Android 8.1'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("OPP28.85-13")', engine='python').index, 'id_30'] = 'Android 8.1' 
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("OPG28.54-19")', engine='python').index, 'id_30'] = 'Android 8.1'   
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("M1AJQ")', engine='python').index, 'id_30'] = 'Android 8.1'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("PPR2.181005.003")', engine='python').index, 'id_30'] = 'Android 9.0'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("PKQ1.180716.001")', engine='python').index, 'id_30'] = 'Android 9.0'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("PPR1.180610.009")', engine='python').index, 'id_30'] = 'Android 9.0'
	    df.loc[df.query('id_30.isnull() and DeviceInfo.str.contains("PPR2.180905.005")', engine='python').index, 'id_30'] = 'Android 9.0'    
	    
	    ########## id_31 ##########
	    # Finding id_31 from DeviceInfo parsing errors
	    df.loc[df.query('id_31.isnull() and DeviceInfo == "rv:52.0"', engine='python').index, 'id_31'] = 'firefox 52.0'
	    
	    ########## id_32 ##########
	    # All iOS devices have 32 bit color depth
	    df.loc[df.query('DeviceInfo == "iOS Device" and id_32.isnull()', engine='python').index, 'id_32'] = 32.0

with faith('4. Fixing UserAgent, id_31 etc and inferring more vals for other cols etc...v1.0') as f:
	for df in tqdm([X_train, X_test]): 
	    ########## DeviceInfo ##########
	    # Fixing DeviceInfo from id_31
	    df.loc[df.query('DeviceInfo == "Windows" and id_31.str.contains("mobile safari")', engine='python').index, 'DeviceInfo'] = 'iOS Device'
	    
	    # Creating a UserAgent feature from DeviceInfo
	    df['UserAgent'] = df['DeviceInfo'].copy()
	    
	    # Fixing DeviceInfo from UserAgent
	    df.loc[df.query('UserAgent == "Intel" and id_30.str.contains("Mac")', engine='python').index, 'DeviceInfo'] = 'MacOS'
	    df.loc[df.query('UserAgent != "MacOS" and id_30.str.startswith("Mac OS")', engine='python').index, 'DeviceInfo'] = 'MacOS'
	    df.loc[df.query('UserAgent == "CPU"').index, 'DeviceInfo'] = 'iOS Device'
	    df.loc[df.query('UserAgent == "Windows NT 6.1"').index, 'DeviceInfo'] = 'Windows'
	    df.loc[df.query('UserAgent == "Windows NT 6.2"').index, 'DeviceInfo'] = 'Windows'
	    df.loc[df.query('UserAgent == "MDDRJS"').index, 'DeviceInfo'] = 'Windows'
	    df.loc[df.query('UserAgent == "Microsoft"').index, 'DeviceInfo'] = 'Windows'
	    df.loc[df.query('UserAgent == "Trident/7.0"').index, 'DeviceInfo'] = 'Windows'
	    df.loc[df.query('UserAgent == "Touch"').index, 'DeviceInfo'] = 'Windows'
	    df.loc[df.query('UserAgent == "S.N.O.W.4"').index, 'DeviceInfo'] = 'Windows'
	    df.loc[df.query('UserAgent == "BOIE9"').index, 'DeviceInfo'] = 'Windows'
	    df.loc[df.query('UserAgent == "rv:11.0"').index, 'DeviceInfo'] = 'Windows'
	    df.loc[df.query('UserAgent == "WOW64"').index, 'DeviceInfo'] = 'Windows'
	    df.loc[df.query('UserAgent == "ATT-IE11"').index, 'DeviceInfo'] = 'Windows'
	    df.loc[df.query('UserAgent == "MAMI"').index, 'DeviceInfo'] = 'Windows'
	    df.loc[df.query('UserAgent == "MALC"').index, 'DeviceInfo'] = 'Windows'
	    df.loc[df.query('UserAgent == "hp2015"').index, 'DeviceInfo'] = 'Windows'
	    df.loc[df.query('UserAgent == "Northwell"').index, 'DeviceInfo'] = 'Windows'
	    df.loc[df.query('UserAgent == "xs-Z47b7VqTMxs"').index, 'DeviceInfo'] = 'Windows'
	    df.loc[df.query('UserAgent == "QwestIE8"').index, 'DeviceInfo'] = 'Windows'
	    df.loc[df.query('UserAgent == "ATT"').index, 'DeviceInfo'] = 'Windows'
	    df.loc[df.query('UserAgent == "NetHelper70"').index, 'DeviceInfo'] = 'Windows'
	    df.loc[df.query('UserAgent == "FunWebProducts"').index, 'DeviceInfo'] = 'Windows'
	    df.loc[df.query('UserAgent == "Lifesize"').index, 'DeviceInfo'] = 'Windows'
	    
	    # Fixing DeviceInfo from id_30
	    df.loc[df.query('id_30.str.startswith("Windows") and UserAgent == "rv:27.0"', engine='python').index, 'DeviceInfo'] = 'Windows'
	    df.loc[df.query('id_30.str.startswith("Windows") and UserAgent == "rv:31.0"', engine='python').index, 'DeviceInfo'] = 'Windows'
	    df.loc[df.query('id_30.str.startswith("Windows") and UserAgent == "rv:37.0"', engine='python').index, 'DeviceInfo'] = 'Windows'
	    df.loc[df.query('id_30.str.startswith("Windows") and UserAgent == "rv:38.0"', engine='python').index, 'DeviceInfo'] = 'Windows'
	    df.loc[df.query('id_30.str.startswith("Windows") and UserAgent == "rv:39.0"', engine='python').index, 'DeviceInfo'] = 'Windows'
	    df.loc[df.query('id_30.str.startswith("Windows") and UserAgent == "rv:42.0"', engine='python').index, 'DeviceInfo'] = 'Windows'
	    df.loc[df.query('id_30.str.startswith("Windows") and UserAgent == "rv:43.0"', engine='python').index, 'DeviceInfo'] = 'Windows'
	    df.loc[df.query('id_30.str.startswith("Windows") and UserAgent == "rv:44.0"', engine='python').index, 'DeviceInfo'] = 'Windows'
	    df.loc[df.query('id_30.str.startswith("Windows") and UserAgent == "rv:45.0"', engine='python').index, 'DeviceInfo'] = 'Windows'
	    df.loc[df.query('id_30.str.startswith("Windows") and UserAgent == "rv:46.0"', engine='python').index, 'DeviceInfo'] = 'Windows'
	    df.loc[df.query('id_30.str.startswith("Windows") and UserAgent == "rv:47.0"', engine='python').index, 'DeviceInfo'] = 'Windows'
	    df.loc[df.query('id_30.str.startswith("Windows") and UserAgent == "rv:48.0"', engine='python').index, 'DeviceInfo'] = 'Windows'
	    df.loc[df.query('id_30.str.startswith("Mac") and UserAgent == "rv:48.0"', engine='python').index, 'DeviceInfo'] = 'MacOS'
	    df.loc[df.query('id_30.str.startswith("Windows") and UserAgent == "rv:49.0"', engine='python').index, 'DeviceInfo'] = 'Windows'
	    df.loc[df.query('id_30.str.startswith("Windows") and UserAgent == "rv:50.0"', engine='python').index, 'DeviceInfo'] = 'Windows'
	    df.loc[df.query('id_30.str.startswith("Windows") and UserAgent == "rv:51.0"', engine='python').index, 'DeviceInfo'] = 'Windows'
	    df.loc[df.query('id_30.str.startswith("Windows") and UserAgent == "rv:52.9"', engine='python').index, 'DeviceInfo'] = 'Windows'
	    df.loc[df.query('id_30.str.startswith("Mac OS") and UserAgent == "rv:57.0"', engine='python').index, 'DeviceInfo'] = 'MacOS'
	    df.loc[df.query('id_30.str.startswith("Windows") and UserAgent == "rv:53.0"', engine='python').index, 'DeviceInfo'] = 'Windows'
	    df.loc[df.query('id_30.str.startswith("Windows") and UserAgent == "rv:54.0"', engine='python').index, 'DeviceInfo'] = 'Windows'
	    df.loc[df.query('id_30.str.startswith("Windows") and UserAgent == "rv:55.0"', engine='python').index, 'DeviceInfo'] = 'Windows'
	    df.loc[df.query('id_30.str.startswith("Windows") and UserAgent == "rv:56.0"', engine='python').index, 'DeviceInfo'] = 'Windows'
	    df.loc[df.query('id_30.str.startswith("Windows") and UserAgent == "rv:57.0"', engine='python').index, 'DeviceInfo'] = 'Windows'
	    df.loc[df.query('id_30.str.startswith("Windows") and UserAgent == "rv:52.0"', engine='python').index, 'DeviceInfo'] = 'Windows'
	    df.loc[df.query('id_30.str.startswith("Windows") and UserAgent == "rv:58.0"', engine='python').index, 'DeviceInfo'] = 'Windows'
	    df.loc[df.query('id_30.str.startswith("Windows") and UserAgent == "rv:60.0"', engine='python').index, 'DeviceInfo'] = 'Windows'
	    df.loc[df.query('id_30.str.startswith("Windows") and UserAgent == "rv:61.0"', engine='python').index, 'DeviceInfo'] = 'Windows'
	    df.loc[df.query('id_30.str.startswith("Windows") and UserAgent == "rv:62.0"', engine='python').index, 'DeviceInfo'] = 'Windows'
	    df.loc[df.query('id_30.str.startswith("Windows") and UserAgent == "rv:63.0"', engine='python').index, 'DeviceInfo'] = 'Windows'
	    df.loc[df.query('id_30.str.startswith("Windows") and UserAgent == "rv:64.0"', engine='python').index, 'DeviceInfo'] = 'Windows'
	    
	    # Incorrect DeviceInfo that can't be found are assigned with NaN
	    df.loc[df.query('DeviceInfo == "rv:14.0"').index, 'DeviceInfo'] = np.nan
	    df.loc[df.query('DeviceInfo == "rv:21.0"').index, 'DeviceInfo'] = np.nan
	    df.loc[df.query('DeviceInfo == "rv:27.0"').index, 'DeviceInfo'] = np.nan
	    df.loc[df.query('DeviceInfo == "rv:28.0"').index, 'DeviceInfo'] = np.nan
	    df.loc[df.query('DeviceInfo == "rv:29.0"').index, 'DeviceInfo'] = np.nan
	    df.loc[df.query('DeviceInfo == "rv:31.0"').index, 'DeviceInfo'] = np.nan
	    df.loc[df.query('DeviceInfo == "rv:33.0"').index, 'DeviceInfo'] = np.nan
	    df.loc[df.query('DeviceInfo == "rv:35.0"').index, 'DeviceInfo'] = np.nan
	    df.loc[df.query('DeviceInfo == "rv:37.0"').index, 'DeviceInfo'] = np.nan
	    df.loc[df.query('DeviceInfo == "rv:38.0"').index, 'DeviceInfo'] = np.nan
	    df.loc[df.query('DeviceInfo == "rv:39.0"').index, 'DeviceInfo'] = np.nan
	    df.loc[df.query('DeviceInfo == "rv:41.0"').index, 'DeviceInfo'] = np.nan
	    df.loc[df.query('DeviceInfo == "rv:42.0"').index, 'DeviceInfo'] = np.nan
	    df.loc[df.query('DeviceInfo == "rv:43.0"').index, 'DeviceInfo'] = np.nan
	    df.loc[df.query('DeviceInfo == "rv:44.0"').index, 'DeviceInfo'] = np.nan
	    df.loc[df.query('DeviceInfo == "rv:45.0"').index, 'DeviceInfo'] = np.nan
	    df.loc[df.query('DeviceInfo == "rv:46.0"').index, 'DeviceInfo'] = np.nan
	    df.loc[df.query('DeviceInfo == "rv:47.0"').index, 'DeviceInfo'] = np.nan
	    df.loc[df.query('DeviceInfo == "rv:48.0"').index, 'DeviceInfo'] = np.nan
	    df.loc[df.query('DeviceInfo == "rv:49.0"').index, 'DeviceInfo'] = np.nan
	    df.loc[df.query('DeviceInfo == "rv:50.0"').index, 'DeviceInfo'] = np.nan
	    df.loc[df.query('DeviceInfo == "rv:51.0"').index, 'DeviceInfo'] = np.nan
	    df.loc[df.query('DeviceInfo == "rv:52.0"').index, 'DeviceInfo'] = np.nan
	    df.loc[df.query('DeviceInfo == "rv:52.9"').index, 'DeviceInfo'] = np.nan
	    df.loc[df.query('DeviceInfo == "rv:53.0"').index, 'DeviceInfo'] = np.nan
	    df.loc[df.query('DeviceInfo == "rv:54.0"').index, 'DeviceInfo'] = np.nan
	    df.loc[df.query('DeviceInfo == "rv:55.0"').index, 'DeviceInfo'] = np.nan
	    df.loc[df.query('DeviceInfo == "rv:56.0"').index, 'DeviceInfo'] = np.nan
	    df.loc[df.query('DeviceInfo == "rv:57.0"').index, 'DeviceInfo'] = np.nan
	    df.loc[df.query('DeviceInfo == "rv:58.0"').index, 'DeviceInfo'] = np.nan
	    df.loc[df.query('DeviceInfo == "rv:59.0"').index, 'DeviceInfo'] = np.nan
	    df.loc[df.query('DeviceInfo == "rv:60.0"').index, 'DeviceInfo'] = np.nan
	    df.loc[df.query('DeviceInfo == "rv:60.1.0"').index, 'DeviceInfo'] = np.nan
	    df.loc[df.query('DeviceInfo == "rv:61.0"').index, 'DeviceInfo'] = np.nan
	    df.loc[df.query('DeviceInfo == "rv:62.0"').index, 'DeviceInfo'] = np.nan
	    df.loc[df.query('DeviceInfo == "rv:63.0"').index, 'DeviceInfo'] = np.nan
	    df.loc[df.query('DeviceInfo == "rv:64.0"').index, 'DeviceInfo'] = np.nan
	    df.loc[df.query('DeviceInfo == "rv:65.0"').index, 'DeviceInfo'] = np.nan
	    df.loc[df.query('DeviceInfo == "en-us"').index, 'DeviceInfo'] = np.nan
	    df.loc[df.query('DeviceInfo == "es-us"').index, 'DeviceInfo'] = np.nan
	    df.loc[df.query('DeviceInfo == "es-es"').index, 'DeviceInfo'] = np.nan
	    df.loc[df.query('DeviceInfo == "es-mx"').index, 'DeviceInfo'] = np.nan
	    df.loc[df.query('DeviceInfo == "en-gb"').index, 'DeviceInfo'] = np.nan
	    df.loc[df.query('DeviceInfo == "Linux"').index, 'DeviceInfo'] = np.nan
	    df.loc[df.query('DeviceInfo == "Android"').index, 'DeviceInfo'] = np.nan
	    df.loc[df.query('DeviceInfo == "Android 4.4.2"').index, 'DeviceInfo'] = np.nan
	    df.loc[df.query('DeviceInfo == "Android 5.1"').index, 'DeviceInfo'] = np.nan
	    df.loc[df.query('DeviceInfo == "Android 6.0.1"').index, 'DeviceInfo'] = np.nan
	    df.loc[df.query('DeviceInfo == "Android 6.0"').index, 'DeviceInfo'] = np.nan
	    df.loc[df.query('DeviceInfo == "Android 7.0"').index, 'DeviceInfo'] = np.nan
	    df.loc[df.query('DeviceInfo == "Android 7.1.2"').index, 'DeviceInfo'] = np.nan
	    df.loc[df.query('DeviceInfo == "Android 8.0.0"').index, 'DeviceInfo'] = np.nan    
	        
	    ########## DeviceType ##########
	    # Fixing DeviceType from UserAgent
	    df.loc[df.query('UserAgent == "Android 4.4.2" and DeviceType == "desktop"').index, 'DeviceType'] = 'mobile'
	    
	    # Fixing DeviceType from DeviceInfo
	    df.loc[df.query('DeviceInfo.str.contains("Build") and DeviceType == "desktop"', engine='python').index, 'DeviceType'] = 'mobile'
	           
	    ########## id_27 ##########
	    # id_27 is the flag which becomes True (Found) when id_23 is not NaN
	    # It is either "Found" for NaN, there is no such value as "NotFound"
	    df.loc[df.query('id_27 == "NotFound"').index, 'id_27'] = 'Found'
	    
	    ########## id_30 ##########
	    
	    # Fixing id_30 from DeviceInfo (Android Build Number)
	    df.loc[df.query('DeviceInfo == "LG-TP260 Build/NRD90U" and id_30 == "func"').index, 'id_30'] = 'Android 7.0'    
	    df.loc[df.query('DeviceInfo.str.contains("IML74K") and id_30 == "Android"', engine='python').index, 'id_30'] = 'Android 4.0' 
	    df.loc[df.query('DeviceInfo.str.contains("JDQ39") and id_30 == "Android"', engine='python').index, 'id_30'] = 'Android 4.2'    
	    df.loc[df.query('DeviceInfo.str.contains("KTU84M") and id_30 == "Android"', engine='python').index, 'id_30'] = 'Android 4.4' 
	    df.loc[df.query('DeviceInfo.str.contains("KTU84P") and id_30 == "Android"', engine='python').index, 'id_30'] = 'Android 4.4'
	    df.loc[df.query('DeviceInfo.str.contains("SU6-7.7") and id_30 == "Android"', engine='python').index, 'id_30'] = 'Android 4.4'    
	    df.loc[df.query('DeviceInfo.str.contains("LRX22C") and id_30 == "Android"', engine='python').index, 'id_30'] = 'Android 5.0'
	    df.loc[df.query('DeviceInfo.str.contains("LMY47D") and id_30 == "Android"', engine='python').index, 'id_30'] = 'Android 5.1'
	    df.loc[df.query('DeviceInfo.str.contains("LMY47O") and id_30 == "Android"', engine='python').index, 'id_30'] = 'Android 5.1'
	    df.loc[df.query('DeviceInfo.str.contains("OPM2.171026.006.G1") and id_30 == "func"', engine='python').index, 'id_30'] = 'Android 8.1'    
	    
	    # Fixing id_30 from UserAgent
	    df.loc[df.query('UserAgent == "Microsoft" and id_30 == "Windows"').index, 'id_30'] = 'Windows 10'       
	    df.loc[df.query('UserAgent == "MDDRJS" and id_30 == "Windows 10"').index, 'id_30'] = 'Windows 7'
	    df.loc[df.query('UserAgent == "Linux"').index, 'id_30'] = 'Linux'
	    df.loc[df.query('UserAgent == "rv:51.0" and id_30 == "Linux"').index, 'id_30'] = 'Android 7.0'
	    df.loc[df.query('UserAgent == "Android" and id_30 == "Windows"').index, 'id_30'] = 'Android'
	    
	    # Fixing id_30 from id_31
	    df.loc[df.query('id_31.str.startswith("edge") and id_30 != "Windows 10"', engine='python').index, 'id_30'] = 'Windows 10' 
	    
	    # Incorrect id_30 that can't be found are assigned with NaN
	    df.loc[df.query('id_31 == "safari" and id_30 == "Android"').index, 'id_30'] = np.nan
	    df.loc[df.query('DeviceInfo.isnull() and id_30 == "other"', engine='python').index, 'id_30'] = np.nan
	    df.loc[df.query('DeviceInfo.isnull() and id_30 == "func"', engine='python').index, 'id_30'] = np.nan
	    
	    # Fixing "other" and "func" id_30 values
	    df.loc[df.query('DeviceInfo == "Windows" and id_30 == "other"').index, 'id_30'] = 'Windows'
	    df.loc[df.query('DeviceInfo == "iOS Device" and id_30 == "other"').index, 'id_30'] = 'iOS'    
	    df.loc[df.query('DeviceInfo == "Windows" and id_30 == "func"').index, 'id_30'] = 'Windows'
	    df.loc[df.query('DeviceInfo == "iOS Device" and id_30 == "func"').index, 'id_30'] = 'iOS'
	    df.loc[df.query('DeviceInfo == "MacOS" and id_30 == "func"').index, 'id_30'] = 'Mac'    
	    
	    # Grouping Android versions
	    df.loc[df.query('id_30 == "Android 4.4.2"').index, 'id_30'] = 'Android 4.4'
	    df.loc[df.query('id_30 == "Android 5.0" or id_30 == "Android 5.0.2"').index, 'id_30'] = 'Android 5.0'
	    df.loc[df.query('id_30 == "Android 5.1" or id_30 == "Android 5.1.1"').index, 'id_30'] = 'Android 5.1'
	    df.loc[df.query('id_30 == "Android 6.0" or id_30 == "Android 6.0.1"').index, 'id_30'] = 'Android 6.0'
	    df.loc[df.query('id_30 == "Android 7.1.1" or id_30 == "Android 7.1.2"').index, 'id_30'] = 'Android 7.1'
	    df.loc[df.query('id_30 == "Android 8.0.0"').index, 'id_30'] = 'Android 8.0'
	    df.loc[df.query('id_30 == "Android 8.1.0"').index, 'id_30'] = 'Android 8.1'
	    df.loc[df.query('id_30 == "Android 9"').index, 'id_30'] = 'Android 9.0'
	    
	    # Grouping Mac OS X versions
	    df.loc[df.query('~id_30.isnull() and (id_30.str.startswith("Mac OS X 10_14") or id_30.str.startswith("Mac OS X 10.14"))', engine='python').index, 'id_30'] = 'Mac OS X 10.14'
	    df.loc[df.query('~id_30.isnull() and (id_30.str.startswith("Mac OS X 10_13") or id_30.str.startswith("Mac OS X 10.13"))', engine='python').index, 'id_30'] = 'Mac OS X 10.13'
	    df.loc[df.query('~id_30.isnull() and (id_30.str.startswith("Mac OS X 10_12") or id_30.str.startswith("Mac OS X 10.12"))', engine='python').index, 'id_30'] = 'Mac OS X 10.12'
	    df.loc[df.query('~id_30.isnull() and (id_30.str.startswith("Mac OS X 10_11") or id_30.str.startswith("Mac OS X 10.11"))', engine='python').index, 'id_30'] = 'Mac OS X 10.11'
	    df.loc[df.query('~id_30.isnull() and (id_30.str.startswith("Mac OS X 10_10") or id_30.str.startswith("Mac OS X 10.10"))', engine='python').index, 'id_30'] = 'Mac OS X 10.10'
	    df.loc[df.query('~id_30.isnull() and (id_30.str.startswith("Mac OS X 10_9") or id_30.str.startswith("Mac OS X 10.9"))', engine='python').index, 'id_30'] = 'Mac OS X 10.9'
	    df.loc[df.query('~id_30.isnull() and (id_30.str.startswith("Mac OS X 10_8") or id_30.str.startswith("Mac OS X 10.8"))', engine='python').index, 'id_30'] = 'Mac OS X 10.8'
	    df.loc[df.query('~id_30.isnull() and (id_30.str.startswith("Mac OS X 10_7") or id_30.str.startswith("Mac OS X 10.7"))', engine='python').index, 'id_30'] = 'Mac OS X 10.7'
	    df.loc[df.query('~id_30.isnull() and (id_30.str.startswith("Mac OS X 10_6") or id_30.str.startswith("Mac OS X 10.6"))', engine='python').index, 'id_30'] = 'Mac OS X 10.6'    
	    
	    ########## id_31 ##########
	    # Fixing id_31 from UserAgent
	    df.loc[df.query('UserAgent == "rv:14.0" and id_31 == "Mozilla/Firefox"').index, 'id_31'] = 'firefox 14.0'
	    df.loc[df.query('UserAgent == "rv:21.0" and id_31 == "firefox"').index, 'id_31'] = 'firefox 21.0'
	    df.loc[df.query('UserAgent == "rv:27.0" and id_31 == "firefox"').index, 'id_31'] = 'firefox 27.0'
	    df.loc[df.query('UserAgent == "rv:28.0" and id_31.isnull()', engine='python').index, 'id_31'] = 'firefox 28.0'
	    df.loc[df.query('UserAgent == "rv:29.0" and id_31 == "firefox"').index, 'id_31'] = 'firefox 29.0'
	    df.loc[df.query('UserAgent == "rv:31.0" and id_31 == "firefox"').index, 'id_31'] = 'firefox 31.0'
	    df.loc[df.query('UserAgent == "rv:33.0" and id_31 == "firefox"').index, 'id_31'] = 'firefox 33.0'
	    df.loc[df.query('UserAgent == "rv:35.0" and id_31 == "firefox"').index, 'id_31'] = 'firefox 35.0'
	    df.loc[df.query('UserAgent == "rv:37.0" and id_31 == "firefox"').index, 'id_31'] = 'firefox 37.0'
	    df.loc[df.query('UserAgent == "rv:38.0" and id_31 == "firefox"').index, 'id_31'] = 'firefox 38.0'
	    df.loc[df.query('UserAgent == "rv:39.0" and id_31 == "firefox"').index, 'id_31'] = 'firefox 39.0'
	    df.loc[df.query('UserAgent == "rv:41.0" and id_31 == "firefox"').index, 'id_31'] = 'firefox 41.0'
	    df.loc[df.query('UserAgent == "rv:42.0" and id_31 == "firefox"').index, 'id_31'] = 'firefox 42.0'
	    df.loc[df.query('UserAgent == "rv:43.0" and id_31 == "firefox"').index, 'id_31'] = 'firefox 43.0'
	    df.loc[df.query('UserAgent == "rv:44.0" and id_31 == "firefox"').index, 'id_31'] = 'firefox 44.0'
	    df.loc[df.query('UserAgent == "rv:45.0" and id_31 == "firefox"').index, 'id_31'] = 'firefox 45.0'
	    df.loc[df.query('UserAgent == "rv:46.0" and id_31 == "firefox"').index, 'id_31'] = 'firefox 46.0'
	    df.loc[df.query('UserAgent == "rv:48.0" and id_31 == "firefox"').index, 'id_31'] = 'firefox 48.0'
	    df.loc[df.query('UserAgent == "rv:49.0" and id_31 == "firefox"').index, 'id_31'] = 'firefox 49.0'
	    df.loc[df.query('UserAgent == "rv:50.0" and id_31 == "firefox"').index, 'id_31'] = 'firefox 50.0'
	    df.loc[df.query('UserAgent == "rv:51.0" and id_31 == "Generic/Android 7.0"').index, 'id_31'] = 'firefox 51.0'
	    df.loc[df.query('UserAgent == "rv:51.0" and id_31 == "seamonkey"').index, 'id_31'] = 'firefox 51.0'
	    df.loc[df.query('UserAgent == "rv:51.0" and id_31 == "firefox"').index, 'id_31'] = 'firefox 51.0'
	    df.loc[df.query('UserAgent == "rv:52.0" and id_31 == "firefox"').index, 'id_31'] = 'firefox 52.0'
	    df.loc[df.query('UserAgent == "rv:52.9" and id_31 == "firefox"').index, 'id_31'] = 'other'
	    df.loc[df.query('UserAgent == "rv:52.9" and id_31 == "palemoon"').index, 'id_31'] = 'other'
	    df.loc[df.query('UserAgent == "rv:53.0" and id_31 == "firefox"').index, 'id_31'] = 'firefox 53.0'
	    df.loc[df.query('UserAgent == "rv:53.0" and id_31 == "Generic/Android"').index, 'id_31'] = 'firefox 53.0'
	    df.loc[df.query('UserAgent == "rv:54.0" and id_31 == "firefox"').index, 'id_31'] = 'firefox 54.0'
	    df.loc[df.query('UserAgent == "rv:55.0" and id_31 == "firefox"').index, 'id_31'] = 'firefox 55.0'
	    df.loc[df.query('UserAgent == "rv:55.0" and id_31 == "Generic/Android"').index, 'id_31'] = 'firefox 55.0'
	    df.loc[df.query('UserAgent == "rv:57.0" and id_31 == "firefox"').index, 'id_31'] = 'firefox 57.0'
	    df.loc[df.query('UserAgent == "rv:56.0" and id_31 == "Generic/Android"').index, 'id_31'] = 'firefox 56.0'
	    df.loc[df.query('UserAgent == "rv:56.0" and id_31 == "firefox"').index, 'id_31'] = 'firefox 56.0'
	    df.loc[df.query('UserAgent == "rv:57.0" and id_31 == "Generic/Android"').index, 'id_31'] = 'firefox 57.0'
	    df.loc[df.query('UserAgent == "rv:57.0" and id_31 == "Generic/Android 7.0"').index, 'id_31'] = 'firefox 57.0'
	    df.loc[df.query('UserAgent == "rv:58.0" and id_31 == "Generic/Android"').index, 'id_31'] = 'firefox 58.0'
	    df.loc[df.query('UserAgent == "rv:58.0" and id_31 == "Generic/Android 7.0"').index, 'id_31'] = 'firefox 58.0'
	    df.loc[df.query('UserAgent == "rv:58.0" and id_31 == "firefox"').index, 'id_31'] = 'firefox 58.0'
	    df.loc[df.query('UserAgent == "rv:58.0" and id_31 == "firefox generic"').index, 'id_31'] = 'firefox 58.0'
	    df.loc[df.query('UserAgent == "rv:59.0" and id_31 == "firefox"').index, 'id_31'] = 'firefox 59.0'
	    df.loc[df.query('UserAgent == "rv:59.0" and id_31 == "firefox generic"').index, 'id_31'] = 'firefox 59.0'
	    df.loc[df.query('UserAgent == "rv:59.0" and id_31 == "Generic/Android"').index, 'id_31'] = 'firefox 59.0'
	    df.loc[df.query('UserAgent == "rv:60.0" and id_31 == "firefox"').index, 'id_31'] = 'firefox 60.0'
	    df.loc[df.query('UserAgent == "rv:60.0" and id_31 == "firefox generic"').index, 'id_31'] = 'firefox 60.0'
	    df.loc[df.query('UserAgent == "rv:60.0" and id_31 == "Generic/Android"').index, 'id_31'] = 'firefox 60.0'
	    df.loc[df.query('UserAgent == "rv:61.0" and id_31 == "Generic/Android"').index, 'id_31'] = 'firefox 61.0'
	    df.loc[df.query('UserAgent == "rv:61.0" and id_31 == "firefox mobile 61.0"').index, 'id_31'] = 'firefox 61.0'
	    df.loc[df.query('UserAgent == "rv:62.0" and id_31 == "firefox mobile 62.0"').index, 'id_31'] = 'firefox 62.0'
	    df.loc[df.query('UserAgent == "rv:63.0" and id_31 == "firefox mobile 63.0"').index, 'id_31'] = 'firefox 63.0'
	    df.loc[df.query('UserAgent == "rv:64.0" and id_31 == "firefox"').index, 'id_31'] = 'firefox 64.0'
	    df.loc[df.query('UserAgent == "rv:65.0" and id_31 == "firefox"').index, 'id_31'] = 'firefox 65.0'

	    # Fixing id_31 from id_30
	    # Safari is part of the iOS and its version is the same as the iOS, finding safari versions from iOS versions
	    df.loc[df.query('id_30.str.startswith("iOS 9") and id_31 == "mobile safari uiwebview"', engine='python').index, 'id_31'] = 'safari 9.0'
	    df.loc[df.query('id_30.str.startswith("iOS 10") and id_31 == "mobile safari uiwebview"', engine='python').index, 'id_31'] = 'safari 10.0'
	    df.loc[df.query('id_30.str.startswith("iOS 11") and id_31 == "mobile safari uiwebview"', engine='python').index, 'id_31'] = 'safari 11.0'
	    
	    # Grouping mobile and desktop safari browsers
	    df.loc[df.query('id_31 == "mobile safari generic" or id_31 == "safari generic"').index, 'id_31'] = 'safari 0.0'
	    df.loc[df.query('id_31 == "mobile safari 8.0" or id_31 == "safari 8.0"').index, 'id_31'] = 'safari 8.0'
	    df.loc[df.query('id_31 == "mobile safari 9.0" or id_31 == "safari 9.0"').index, 'id_31'] = 'safari 9.0'
	    df.loc[df.query('id_31 == "mobile safari 10.0" or id_31 == "safari 10.0"').index, 'id_31'] = 'safari 10.0'
	    df.loc[df.query('id_31 == "mobile safari 11.0" or id_31 == "safari 11.0"').index, 'id_31'] = 'safari 11.0'
	    df.loc[df.query('id_31 == "mobile safari 12.0" or id_31 == "safari 12.0"').index, 'id_31'] = 'safari 12.0'
	    
	    # Grouping mobile and desktop chrome browsers
	    df.loc[df.query('id_31 == "chrome 39.0 for android"').index, 'id_31'] = 'chrome 39.0'
	    df.loc[df.query('id_31 == "chrome 43.0 for android"').index, 'id_31'] = 'chrome 43.0'
	    df.loc[df.query('id_31 == "chrome 46.0 for android"').index, 'id_31'] = 'chrome 46.0'
	    df.loc[df.query('id_31 == "google search application 48.0"').index, 'id_31'] = 'chrome 48.0'
	    df.loc[df.query('id_31 == "chrome 49.0 for android" or id_31 == "chrome 49.0" or id_31 == "google search application 49.0"').index, 'id_31'] = 'chrome 49.0'
	    df.loc[df.query('id_31 == "chrome 50.0 for android"').index, 'id_31'] = 'chrome 50.0'
	    df.loc[df.query('id_31 == "chrome 51.0 for android" or id_31 == "chrome 51.0"').index, 'id_31'] = 'chrome 51.0'
	    df.loc[df.query('id_31 == "chrome 52.0 for android" or id_31 == "google search application 52.0"').index, 'id_31'] = 'chrome 52.0'
	    df.loc[df.query('id_31 == "chrome 53.0 for android"').index, 'id_31'] = 'chrome 53.0'
	    df.loc[df.query('id_31 == "chrome 54.0 for android" or id_31 == "google search application 54.0"').index, 'id_31'] = 'chrome 54.0'
	    df.loc[df.query('id_31 == "chrome 55.0 for android" or id_31 == "chrome 55.0"').index, 'id_31'] = 'chrome 55.0'
	    df.loc[df.query('id_31 == "chrome 56.0 for android" or id_31 == "chrome 56.0" or id_31 == "google search application 56.0"').index, 'id_31'] = 'chrome 56.0'
	    df.loc[df.query('id_31 == "chrome 57.0 for android" or id_31 == "chrome 57.0"').index, 'id_31'] = 'chrome 57.0'
	    df.loc[df.query('id_31 == "chrome 58.0 for android" or id_31 == "chrome 58.0" or id_31 == "google search application 58.0"').index, 'id_31'] = 'chrome 58.0'
	    df.loc[df.query('id_31 == "chrome 59.0 for android" or id_31 == "chrome 59.0" or id_31 == "google search application 59.0"').index, 'id_31'] = 'chrome 59.0'
	    df.loc[df.query('id_31 == "chrome 60.0 for android" or id_31 == "chrome 60.0" or id_31 == "google search application 60.0"').index, 'id_31'] = 'chrome 60.0'
	    df.loc[df.query('id_31 == "chrome 61.0 for android" or id_31 == "chrome 61.0" or id_31 == "google search application 61.0"').index, 'id_31'] = 'chrome 61.0'
	    df.loc[df.query('id_31 == "chrome 62.0 for android" or id_31 == "chrome 62.0 for ios" or id_31 == "chrome 62.0" or id_31 == "google search application 62.0"').index, 'id_31'] = 'chrome 62.0'
	    df.loc[df.query('id_31 == "chrome 63.0 for android" or id_31 == "chrome 63.0 for ios" or id_31 == "chrome 63.0" or id_31 == "google search application 63.0"').index, 'id_31'] = 'chrome 63.0'
	    df.loc[df.query('id_31 == "chrome 64.0 for android" or id_31 == "chrome 64.0 for ios" or id_31 == "chrome 64.0" or id_31 == "google search application 64.0"').index, 'id_31'] = 'chrome 64.0'
	    df.loc[df.query('id_31 == "chrome 65.0 for android" or id_31 == "chrome 65.0 for ios" or id_31 == "chrome 65.0" or id_31 == "google search application 65.0"').index, 'id_31'] = 'chrome 65.0'
	    df.loc[df.query('id_31 == "chrome 66.0 for android" or id_31 == "chrome 66.0 for ios" or id_31 == "chrome 66.0"').index, 'id_31'] = 'chrome 66.0'
	    df.loc[df.query('id_31 == "chrome 67.0 for android" or id_31 == "chrome 67.0 for ios" or id_31 == "chrome 67.0"').index, 'id_31'] = 'chrome 67.0'
	    df.loc[df.query('id_31 == "chrome 68.0 for android" or id_31 == "chrome 68.0 for ios" or id_31 == "chrome 68.0"').index, 'id_31'] = 'chrome 68.0'
	    df.loc[df.query('id_31 == "chrome 69.0 for android" or id_31 == "chrome 69.0 for ios" or id_31 == "chrome 69.0"').index, 'id_31'] = 'chrome 69.0'
	    df.loc[df.query('id_31 == "chrome 70.0 for android" or id_31 == "chrome 70.0 for ios" or id_31 == "chrome 70.0"').index, 'id_31'] = 'chrome 70.0'
	    df.loc[df.query('id_31 == "chrome 71.0 for android" or id_31 == "chrome 71.0 for ios" or id_31 == "chrome 71.0"').index, 'id_31'] = 'chrome 71.0'
	    
	    # Grouping mobile and desktop firefox browsers
	    df.loc[df.query('id_31 == "firefox mobile 61.0" or id_31 == "firefox 61.0"').index, 'id_31'] = 'firefox 61.0'
	    df.loc[df.query('id_31 == "firefox mobile 62.0" or id_31 == "firefox 62.0"').index, 'id_31'] = 'firefox 62.0'
	    df.loc[df.query('id_31 == "firefox mobile 63.0" or id_31 == "firefox 63.0"').index, 'id_31'] = 'firefox 63.0'
	    
	    # Grouping other id_31 values
	    df.loc[df.query('id_31 == "Samsung/SM-G532M"').index, 'id_31'] = 'other'
	    df.loc[df.query('id_31 == "Samsung/SM-G531H"').index, 'id_31'] = 'other'  
	    df.loc[df.query('id_31 == "Generic/Android 7.0"').index, 'id_31'] = 'other'
	    df.loc[df.query('id_31 == "palemoon"').index, 'id_31'] = 'other'
	    df.loc[df.query('id_31 == "cyberfox"').index, 'id_31'] = 'other'
	    df.loc[df.query('id_31 == "android"').index, 'id_31'] = 'other'
	    df.loc[df.query('id_31 == "Cherry"').index, 'id_31'] = 'other'
	    df.loc[df.query('id_31 == "M4Tel/M4"').index, 'id_31'] = 'other'
	    df.loc[df.query('id_31 == "Samsung/SCH"').index, 'id_31'] = 'other'
	    df.loc[df.query('id_31 == "chromium"').index, 'id_31'] = 'other'
	    df.loc[df.query('id_31 == "BLU/Dash"').index, 'id_31'] = 'other'
	    df.loc[df.query('id_31 == "Nokia/Lumia"').index, 'id_31'] = 'other'
	    df.loc[df.query('id_31 == "LG/K-200"').index, 'id_31'] = 'other'
	    df.loc[df.query('id_31 == "iron"').index, 'id_31'] = 'other'
	    df.loc[df.query('id_31 == "Inco/Minion"').index, 'id_31'] = 'other'
	    df.loc[df.query('id_31 == "waterfox"').index, 'id_31'] = 'other'
	    df.loc[df.query('id_31 == "facebook"').index, 'id_31'] = 'other'
	    df.loc[df.query('id_31 == "puffin"').index, 'id_31'] = 'other'
	    df.loc[df.query('id_31 == "Lanix/Ilium"').index, 'id_31'] = 'other'
	    df.loc[df.query('id_31 == "icedragon"').index, 'id_31'] = 'other'
	    df.loc[df.query('id_31 == "aol"').index, 'id_31'] = 'other'
	    df.loc[df.query('id_31 == "comodo"').index, 'id_31'] = 'other'
	    df.loc[df.query('id_31 == "line"').index, 'id_31'] = 'other'
	    df.loc[df.query('id_31 == "maxthon"').index, 'id_31'] = 'other'
	    df.loc[df.query('id_31 == "ZTE/Blade"').index, 'id_31'] = 'other'
	    df.loc[df.query('id_31 == "mobile"').index, 'id_31'] = 'other'
	    df.loc[df.query('id_31 == "silk"').index, 'id_31'] = 'other'
	    df.loc[df.query('id_31 == "Microsoft/Windows"').index, 'id_31'] = 'other'
	    df.loc[df.query('id_31 == "rim"').index, 'id_31'] = 'other'
	    df.loc[df.query('id_31 == "blackberry"').index, 'id_31'] = 'other'
	    df.loc[df.query('id_31 == "uc"').index, 'id_31'] = 'other'
	   
	    ########## UserAgent ##########
	    # Grouping rv:60 UserAgent values
	    df.loc[df.query('UserAgent == "rv:60.1.0"').index, 'UserAgent'] = 'rv:60.0'
	    # Removing DeviceInfo values from UserAgent
	    df.loc[df.query('~UserAgent.isin(@USER_AGENTS) and ~UserAgent.isnull()', engine='python').index, 'UserAgent'] = np.nan
	    
	    ########## id_32 ##########
	    # 0.0 color depth is fixed with the mode value
	    df.loc[df.query('id_32 == 0.0 and UserAgent == "rv:59.0"').index, 'id_32'] = 24.0

with faith('5. Fixing UserAgent, id_31 etc and inferring more vals for other cols etc... v2.0, Yes it\'s twice') as f:
	for df in tqdm([X_train, X_test]):
	        
	    # DeviceInfo
	    df.loc[df.query('DeviceInfo.str.contains("SM-J700M") and ~DeviceInfo.isnull()', engine='python').index, 'DeviceInfo'] = 'Samsung-J700M'
	    df.loc[df.query('DeviceInfo.str.contains("SM-G610M") and ~DeviceInfo.isnull()', engine='python').index, 'DeviceInfo'] = 'Samsung-G610M'
	    df.loc[df.query('DeviceInfo.str.contains("SM-G531H") and ~DeviceInfo.isnull()', engine='python').index, 'DeviceInfo'] = 'Samsung-G531H'
	    df.loc[df.query('DeviceInfo.str.contains("SM-G935F") and ~DeviceInfo.isnull()', engine='python').index, 'DeviceInfo'] = 'Samsung-G935F'
	    df.loc[df.query('DeviceInfo.str.contains("SM-G955U") and ~DeviceInfo.isnull()', engine='python').index, 'DeviceInfo'] = 'Samsung-G955U'
	    df.loc[df.query('DeviceInfo.str.contains("SM-G532M") and ~DeviceInfo.isnull()', engine='python').index, 'DeviceInfo'] = 'Samsung-G532M'
	    df.loc[df.query('DeviceInfo.str.contains("ALE") and ~DeviceInfo.isnull()', engine='python').index, 'DeviceInfo'] = 'Huawei-ALE'  
	    df.loc[df.query('DeviceInfo.str.contains("SM-G950U") and ~DeviceInfo.isnull()', engine='python').index, 'DeviceInfo'] = 'Samsung-G950U'
	    df.loc[df.query('DeviceInfo.str.contains("SM-G930V") and ~DeviceInfo.isnull()', engine='python').index, 'DeviceInfo'] = 'Samsung-G930V' 
	    df.loc[df.query('DeviceInfo.str.contains("SM-G950F") and ~DeviceInfo.isnull()', engine='python').index, 'DeviceInfo'] = 'Samsung-G950F'
	    df.loc[df.query('DeviceInfo.str.contains("Moto G \(4\)") and ~DeviceInfo.isnull()', engine='python').index, 'DeviceInfo'] = 'Motorola-Moto-G4'
	    df.loc[df.query('DeviceInfo.str.contains("SM-N950U") and ~DeviceInfo.isnull()', engine='python').index, 'DeviceInfo'] = 'Samsung-N950U'
	    df.loc[df.query('DeviceInfo.str.contains("SM-A300H") and ~DeviceInfo.isnull()', engine='python').index, 'DeviceInfo'] = 'Samsung-A300H'
	    df.loc[df.query('DeviceInfo.str.contains("hi6210sft") and ~DeviceInfo.isnull()', engine='python').index, 'DeviceInfo'] = 'Huawei-hi6210sft'
	    df.loc[df.query('DeviceInfo.str.contains("SM-J730GM") and ~DeviceInfo.isnull()', engine='python').index, 'DeviceInfo'] = 'Samsung-J730GM'
	    df.loc[df.query('DeviceInfo.str.contains("SM-G570M") and ~DeviceInfo.isnull()', engine='python').index, 'DeviceInfo'] = 'Samsung-G570M'
	    df.loc[df.query('DeviceInfo.str.contains("CAM-") and ~DeviceInfo.isnull()', engine='python').index, 'DeviceInfo'] = 'Huawei-CAM'
	    df.loc[df.query('DeviceInfo.str.contains("SM-J320M") and ~DeviceInfo.isnull()', engine='python').index, 'DeviceInfo'] = 'Samsung-J320M'
	    df.loc[df.query('DeviceInfo.str.contains("Moto E \(4\) Plus") and ~DeviceInfo.isnull()', engine='python').index, 'DeviceInfo'] = 'Motorola-Moto-E4-Plus'
	    df.loc[df.query('DeviceInfo.str.contains("Moto E \(4\)") and ~DeviceInfo.isnull()', engine='python').index, 'DeviceInfo'] = 'Motorola-Moto-E4'
	    df.loc[df.query('DeviceInfo.str.contains("LG-M700") and ~DeviceInfo.isnull()', engine='python').index, 'DeviceInfo'] = 'LG-M700'
	    df.loc[df.query('DeviceInfo.str.contains("ANE") and ~DeviceInfo.isnull()', engine='python').index, 'DeviceInfo'] = 'Huawei-ANE'    
	    df.loc[df.query('DeviceInfo.str.contains("SM-J510MN") and ~DeviceInfo.isnull()', engine='python').index, 'DeviceInfo'] = 'Samsung-J510MN'
	    df.loc[df.query('DeviceInfo.str.contains("SM-J701M") and ~DeviceInfo.isnull()', engine='python').index, 'DeviceInfo'] = 'Samsung-J701M'
	    df.loc[df.query('DeviceInfo.str.contains("LG-D693n") and ~DeviceInfo.isnull()', engine='python').index, 'DeviceInfo'] = 'LG-D693n'
	    df.loc[df.query('DeviceInfo.str.contains("SM-A520F") and ~DeviceInfo.isnull()', engine='python').index, 'DeviceInfo'] = 'Samsung-A520F'
	    df.loc[df.query('DeviceInfo.str.contains("SM-G930F") and ~DeviceInfo.isnull()', engine='python').index, 'DeviceInfo'] = 'Samsung-G930F'
	    df.loc[df.query('DeviceInfo.str.contains("SM-G935V") and ~DeviceInfo.isnull()', engine='python').index, 'DeviceInfo'] = 'Samsung-G935V'
	    df.loc[df.query('DeviceInfo.str.contains("LG-K410") and ~DeviceInfo.isnull()', engine='python').index, 'DeviceInfo'] = 'LG-K410'
	    df.loc[df.query('DeviceInfo.str.contains("PRA-") and ~DeviceInfo.isnull()', engine='python').index, 'DeviceInfo'] = 'Huawei-PRA'
	    df.loc[df.query('DeviceInfo.str.contains("SM-G955F") and ~DeviceInfo.isnull()', engine='python').index, 'DeviceInfo'] = 'Samsung-G955F'
	    df.loc[df.query('DeviceInfo.str.contains("Moto G \(5\) Plus") and ~DeviceInfo.isnull()', engine='python').index, 'DeviceInfo'] = 'Motorola-Moto-G5-Plus'
	    df.loc[df.query('DeviceInfo.str.contains("Moto G \(5\)") and ~DeviceInfo.isnull()', engine='python').index, 'DeviceInfo'] = 'Motorola-Moto-G5'
	    df.loc[df.query('DeviceInfo.str.contains("Moto Z2") and ~DeviceInfo.isnull()', engine='python').index, 'DeviceInfo'] = 'Motorola-Moto-Z2-Play'
	    df.loc[df.query('DeviceInfo.str.contains("TRT-") and ~DeviceInfo.isnull()', engine='python').index, 'DeviceInfo'] = 'Huawei-TRT'
	    df.loc[df.query('DeviceInfo.str.contains("Moto G Play") and ~DeviceInfo.isnull()', engine='python').index, 'DeviceInfo'] = 'Motorola-Moto-G-Play'
	    df.loc[df.query('DeviceInfo.str.contains("SM-A720F") and ~DeviceInfo.isnull()', engine='python').index, 'DeviceInfo'] = 'Samsung-A720F'
	    df.loc[df.query('DeviceInfo.str.contains("LG-K580") and ~DeviceInfo.isnull()', engine='python').index, 'DeviceInfo'] = 'LG-K580'
	    df.loc[df.query('DeviceInfo.str.contains("TAG-") and ~DeviceInfo.isnull()', engine='python').index, 'DeviceInfo'] = 'Huawei-TAG'
	    df.loc[df.query('DeviceInfo.str.contains("VNS-") and ~DeviceInfo.isnull()', engine='python').index, 'DeviceInfo'] = 'Huawei-VNS'   
	    df.loc[df.query('DeviceInfo.str.contains("Moto X Play") and ~DeviceInfo.isnull()', engine='python').index, 'DeviceInfo'] = 'Motorola-Moto-X-Play'
	    df.loc[df.query('DeviceInfo.str.contains("LG-X230") and ~DeviceInfo.isnull()', engine='python').index, 'DeviceInfo'] = 'LG-X230'
	    df.loc[df.query('DeviceInfo.str.contains("WAS-") and ~DeviceInfo.isnull()', engine='python').index, 'DeviceInfo'] = 'Huawei-WAS'
	    
	    # id_30
	    df.loc[df.query('DeviceInfo == "Samsung-J700M" and id_30.isnull()', engine='python').index, 'id_30'] = 'Android'
	    df.loc[df.query('DeviceInfo == "Samsung-G610M" and id_30.isnull()', engine='python').index, 'id_30'] = 'Android'
	    df.loc[df.query('DeviceInfo == "Samsung-G531H" and id_30.isnull()', engine='python').index, 'id_30'] = 'Android'
	    df.loc[df.query('DeviceInfo == "Samsung-G935F" and id_30.isnull()', engine='python').index, 'id_30'] = 'Android'
	    df.loc[df.query('DeviceInfo == "Samsung-G955U" and id_30.isnull()', engine='python').index, 'id_30'] = 'Android'
	    df.loc[df.query('DeviceInfo == "Samsung-G532M" and id_30.isnull()', engine='python').index, 'id_30'] = 'Android'
	    df.loc[df.query('DeviceInfo == "Huawei-ALE" and id_30.isnull()', engine='python').index, 'id_30'] = 'Android'
	    df.loc[df.query('DeviceInfo == "Samsung-G950U" and id_30.isnull()', engine='python').index, 'id_30'] = 'Android'
	    df.loc[df.query('DeviceInfo == "Samsung-G930V" and id_30.isnull()', engine='python').index, 'id_30'] = 'Android'
	    df.loc[df.query('DeviceInfo == "Samsung-G950F" and id_30.isnull()', engine='python').index, 'id_30'] = 'Android'
	    df.loc[df.query('DeviceInfo == "Motorola-Moto-G4" and id_30.isnull()', engine='python').index, 'id_30'] = 'Android'
	    df.loc[df.query('DeviceInfo == "Samsung-N950U" and id_30.isnull()', engine='python').index, 'id_30'] = 'Android'
	    df.loc[df.query('DeviceInfo == "Samsung-A300H" and id_30.isnull()', engine='python').index, 'id_30'] = 'Android'
	    df.loc[df.query('DeviceInfo == "Huawei-hi6210sft" and id_30.isnull()', engine='python').index, 'id_30'] = 'Android'
	    df.loc[df.query('DeviceInfo == "Samsung-J730GM" and id_30.isnull()', engine='python').index, 'id_30'] = 'Android'
	    df.loc[df.query('DeviceInfo == "Samsung-G570M" and id_30.isnull()', engine='python').index, 'id_30'] = 'Android'
	    df.loc[df.query('DeviceInfo == "Huawei-CAM" and id_30.isnull()', engine='python').index, 'id_30'] = 'Android'
	    df.loc[df.query('DeviceInfo == "Samsung-J320M" and id_30.isnull()', engine='python').index, 'id_30'] = 'Android'
	    df.loc[df.query('DeviceInfo == "Motorola-Moto-E4-Plus" and id_30.isnull()', engine='python').index, 'id_30'] = 'Android'
	    df.loc[df.query('DeviceInfo == "Motorola-Moto-E4" and id_30.isnull()', engine='python').index, 'id_30'] = 'Android'
	    df.loc[df.query('DeviceInfo == "LG-M700" and id_30.isnull()', engine='python').index, 'id_30'] = 'Android'
	    df.loc[df.query('DeviceInfo == "Huawei-ANE" and id_30.isnull()', engine='python').index, 'id_30'] = 'Android'
	    df.loc[df.query('DeviceInfo == "Samsung-J510MN" and id_30.isnull()', engine='python').index, 'id_30'] = 'Android'
	    df.loc[df.query('DeviceInfo == "Samsung-J701M" and id_30.isnull()', engine='python').index, 'id_30'] = 'Android'
	    df.loc[df.query('DeviceInfo == "LG-D693n" and id_30.isnull()', engine='python').index, 'id_30'] = 'Android'
	    df.loc[df.query('DeviceInfo == "Samsung-A520F" and id_30.isnull()', engine='python').index, 'id_30'] = 'Android'
	    df.loc[df.query('DeviceInfo == "Samsung-G930F" and id_30.isnull()', engine='python').index, 'id_30'] = 'Android'
	    df.loc[df.query('DeviceInfo == "Samsung-G935V" and id_30.isnull()', engine='python').index, 'id_30'] = 'Android'
	    df.loc[df.query('DeviceInfo == "LG-K410" and id_30.isnull()', engine='python').index, 'id_30'] = 'Android'
	    df.loc[df.query('DeviceInfo == "Huawei-PRA" and id_30.isnull()', engine='python').index, 'id_30'] = 'Android'
	    df.loc[df.query('DeviceInfo == "Samsung-G955F" and id_30.isnull()', engine='python').index, 'id_30'] = 'Android'
	    df.loc[df.query('DeviceInfo == "Motorola-Moto-G5-Plus" and id_30.isnull()', engine='python').index, 'id_30'] = 'Android'
	    df.loc[df.query('DeviceInfo == "Motorola-Moto-G5" and id_30.isnull()', engine='python').index, 'id_30'] = 'Android'
	    df.loc[df.query('DeviceInfo == "Motorola-Moto-Z2-Play" and id_30.isnull()', engine='python').index, 'id_30'] = 'Android'
	    df.loc[df.query('DeviceInfo == "Huawei-TRT" and id_30.isnull()', engine='python').index, 'id_30'] = 'Android'
	    df.loc[df.query('DeviceInfo == "Motorola-Moto-G-Play" and id_30.isnull()', engine='python').index, 'id_30'] = 'Android'
	    df.loc[df.query('DeviceInfo == "Samsung-A720F" and id_30.isnull()', engine='python').index, 'id_30'] = 'Android'
	    df.loc[df.query('DeviceInfo == "LG-K580" and id_30.isnull()', engine='python').index, 'id_30'] = 'Android'
	    df.loc[df.query('DeviceInfo == "Huawei-TAG" and id_30.isnull()', engine='python').index, 'id_30'] = 'Android'
	    df.loc[df.query('DeviceInfo == "Huawei-VNS" and id_30.isnull()', engine='python').index, 'id_30'] = 'Android'
	    df.loc[df.query('DeviceInfo == "Motorola-Moto-X-Play" and id_30.isnull()', engine='python').index, 'id_30'] = 'Android'
	    df.loc[df.query('DeviceInfo == "LG-X230" and id_30.isnull()', engine='python').index, 'id_30'] = 'Android'
	    df.loc[df.query('DeviceInfo == "Huawei-WAS" and id_30.isnull()', engine='python').index, 'id_30'] = 'Android'
	        
	    # id_33
	    df.loc[df.query('DeviceInfo == "Samsung-J700M"').index, 'id_33'] = '1280x720'
	    df.loc[df.query('DeviceInfo == "Samsung-G610M"').index, 'id_33'] = '1920x1080'
	    df.loc[df.query('DeviceInfo == "Samsung-G531H"').index, 'id_33'] = '1280x720'
	    df.loc[df.query('DeviceInfo == "Samsung-G935F"').index, 'id_33'] = '2560x1440'
	    df.loc[df.query('DeviceInfo == "Samsung-G955U"').index, 'id_33'] = '2960x1440'
	    df.loc[df.query('DeviceInfo == "Samsung-G532M"').index, 'id_33'] = '960x540'
	    df.loc[df.query('DeviceInfo == "Huawei-ALE"').index, 'id_33'] = '1280x720'
	    df.loc[df.query('DeviceInfo == "Samsung-G950U"').index, 'id_33'] = '2960x1440'
	    df.loc[df.query('DeviceInfo == "Samsung-G930V"').index, 'id_33'] = '2560x1440'    
	    df.loc[df.query('DeviceInfo == "Samsung-G950F"').index, 'id_33'] = '2960x1440'
	    df.loc[df.query('DeviceInfo == "Motorola-Moto-G4"').index, 'id_33'] = '1920x1080'
	    df.loc[df.query('DeviceInfo == "Samsung-N950U"').index, 'id_33'] = '2960x1440'
	    df.loc[df.query('DeviceInfo == "Samsung-A300H"').index, 'id_33'] = '960x540'
	    df.loc[df.query('DeviceInfo == "Huawei-hi6210sft"').index, 'id_33'] = '1280x720'
	    df.loc[df.query('DeviceInfo == "Samsung-J730GM"').index, 'id_33'] = '1920x1080'
	    df.loc[df.query('DeviceInfo == "Samsung-G570M"').index, 'id_33'] = '1280x720'
	    df.loc[df.query('DeviceInfo == "Huawei-CAM"').index, 'id_33'] = '1280x720'
	    df.loc[df.query('DeviceInfo == "Samsung-J320M"').index, 'id_33'] = '1280x720'
	    df.loc[df.query('DeviceInfo == "Motorola-Moto-E4-Plus"').index, 'id_33'] = '1280x720'
	    df.loc[df.query('DeviceInfo == "Motorola-Moto-E4"').index, 'id_33'] = '1280x720'
	    df.loc[df.query('DeviceInfo == "LG-M700"').index, 'id_33'] = '2880x1440'
	    df.loc[df.query('DeviceInfo == "Huawei-ANE"').index, 'id_33'] = '2280x1080'
	    df.loc[df.query('DeviceInfo == "Samsung-J510MN"').index, 'id_33'] = '1280x720'
	    df.loc[df.query('DeviceInfo == "Samsung-J701M"').index, 'id_33'] = '1280x720'
	    df.loc[df.query('DeviceInfo == "LG-D693n"').index, 'id_33'] = '960x540'
	    df.loc[df.query('DeviceInfo == "Samsung-A520F"').index, 'id_33'] = '960x540'
	    df.loc[df.query('DeviceInfo == "Samsung-G930F"').index, 'id_33'] = '2560x1440'
	    df.loc[df.query('DeviceInfo == "Samsung-G935V"').index, 'id_33'] = '2560x1440'
	    df.loc[df.query('DeviceInfo == "LG-K410"').index, 'id_33'] = '1280x720'
	    df.loc[df.query('DeviceInfo == "Huawei-PRA"').index, 'id_33'] = '1920x1080'
	    df.loc[df.query('DeviceInfo == "Samsung-G955F"').index, 'id_33'] = '2960x1440'
	    df.loc[df.query('DeviceInfo == "Motorola-Moto-G5-Plus"').index, 'id_33'] = '1920x1080'
	    df.loc[df.query('DeviceInfo == "Motorola-Moto-G5"').index, 'id_33'] = '1920x1080'
	    df.loc[df.query('DeviceInfo == "Motorola-Moto-Z2-Play"').index, 'id_33'] = '1920x1080'
	    df.loc[df.query('DeviceInfo == "Huawei-TRT"').index, 'id_33'] = '1280x720'
	    df.loc[df.query('DeviceInfo == "Motorola-Moto-G-Play"').index, 'id_33'] = '1920x1080'
	    df.loc[df.query('DeviceInfo == "Samsung-A720F"').index, 'id_33'] = '1920x1080'
	    df.loc[df.query('DeviceInfo == "LG-K580"').index, 'id_33'] = '1920x1080'
	    df.loc[df.query('DeviceInfo == "Huawei-TAG"').index, 'id_33'] = '1280x720'
	    df.loc[df.query('DeviceInfo == "Huawei-VNS"').index, 'id_33'] = '1280x720'
	    df.loc[df.query('DeviceInfo == "Motorola-Moto-X-Play"').index, 'id_33'] = '1920x1080'
	    df.loc[df.query('DeviceInfo == "LG-X230"').index, 'id_33'] = '854x480'
	    df.loc[df.query('DeviceInfo == "Huawei-WAS"').index, 'id_33'] = '1920x1080'

# In[7]:

D_COLS = [f'D{i}' for i in range(1, 16) if i != 9]
C_COLS = [f'C{i}' for i in range(1, 15)]
CN_COLS = [f'C{i}_Week_Norm' for i in range(1, 15)]

with faith('6. Creating ID cols suing card cols and adding C, D, V blocks norm feats etc') as f:
	for df in tqdm([X_train, X_test]):
		
	    # Feature Combination
	    df['Card_ID1'] = df['card1'].astype(str) + '_' + df['card2'].astype(str)
	    df['Card_ID2'] = df['Card_ID1'].astype(str) + '_' + df['card3'].astype(str) + '_' + df['card5'].astype(str)
	    df['Card_ID3'] = df['Card_ID2'].astype(str) + '_' + df['addr1'].astype(str) + '_' + df['addr2'].astype(str)
	    df['Item_ID'] = df['TransactionAmt'].astype(str) + '_' + df['ProductCD'].astype(str)
 	    #df['Device_ID'] = df['DeviceInfo'].astype(str) + '_' + df['DeviceType'].astype(str) + '_' + df['id_30'].astype(str) + '_' + df['id_31'].astype(str)
	    df['PAccount_ID'] = df['addr1'].astype(str) + '_' + df['addr2'].astype(str) + '_' + df['P_emaildomain'].astype(str)
	    df['RAccount_ID'] = df['addr1'].astype(str) + '_' + df['addr2'].astype(str) + '_' + df['R_emaildomain'].astype(str)
	    df['PR_emaildomain'] = df['P_emaildomain'].astype(str) + '_' + df['R_emaildomain'].astype(str)
	    
	    # D unique count
	    df['D_Uniques'] = df[D_COLS].nunique(axis=1)
	    
	    # D Normalized
	    for d in D_COLS:
	    	df[f'{d}_Week_Norm'] = df[d] - df['Week'].map(
	    		pd.concat([X_train[[d, 'Week']], X_test[[d, 'Week']]], ignore_index=True).groupby('Week')[d].mean()
	    		)
	    
	    # V-Block Aggregation
	    for block in [(1, 12), (12, 35), (35, 53), (53, 75), (75, 95), (95, 138), (138, 167), (167, 217), (217, 279),  (279, 322), (322, 340)]:
	        df['V{}-V{}_Sum'.format(*block)] = df[['V{}'.format(i) for i in range(*block)]].sum(axis=1)
	        df['V{}-V{}_Mean'.format(*block)] = df[['V{}'.format(i) for i in range(*block)]].mean(axis=1)
	        df['V{}-V{}_Std'.format(*block)] = df[['V{}'.format(i) for i in range(*block)]].std(axis=1)

# CONTINUOUS/CATEGORICAL GROUPING AGGREGATIONS
DN_COLS = [f'D{i}_Week_Norm' for i in range(1, 16) if i != 9]
CONT_COLS = ['TransactionAmt', 'dist1', 'dist2'] + C_COLS + DN_COLS
CAT_COLS = ['card1', 'card2', 'card3', 'card5', 'Card_ID1', 'Card_ID2', 'Card_ID3', 'addr1', 'P_emaildomain',
            'R_emaildomain', 'PAccount_ID', 'RAccount_ID', 'PR_emaildomain']
AGG_TYPES = ['std', 'mean', 'sum']

with faith('7. Adding various other agg feats') as f:
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

# Not using for memory issues
'''
CONT_COLS = ['TransactionAmt', 'id_01', 'id_02']
TIME_COLS = ['Hour', 'Day', 'Week']

for df in [X_train, X_test]:
    
    # Continuous - Continuous/Categorical group Mean/Median Difference
    for cont_col in CONT_COLS:
        for cat_col in CAT_COLS:
            df['{}_({}Mean{})_Difference'.format(cont_col, cat_col, cont_col)] = df[cont_col] - df[cat_col].map(pd.concat([X_train[[cont_col, cat_col]], X_test[[cont_col, cat_col]]], ignore_index=True).groupby(cat_col)[cont_col].mean())
            df['{}_({}Median{})_Difference'.format(cont_col, cat_col, cont_col)] = df[cont_col] - df[cat_col].map(pd.concat([X_train[[cont_col, cat_col]], X_test[[cont_col, cat_col]]], ignore_index=True).groupby(cat_col)[cont_col].median())
            
            gc.collect()
            
    # Time-based continuous aggregation
    for cont_col in CONT_COLS:
        for time_col in TIME_COLS:
            df['{}_{}_Sum'.format(time_col, cont_col)] = df[time_col].map(pd.concat([X_train[[cont_col, time_col]], X_test[[cont_col, time_col]]], ignore_index=True).groupby(time_col)[cont_col].sum())
            df['{}_{}_Count'.format(time_col, cont_col)] = df[time_col].map(pd.concat([X_train[[cont_col, time_col]], X_test[[cont_col, time_col]]], ignore_index=True).groupby(time_col)[cont_col].count())
            df['{}_{}_Mean'.format(time_col, cont_col)] = df[time_col].map(pd.concat([X_train[[cont_col, time_col]], X_test[[cont_col, time_col]]], ignore_index=True).groupby(time_col)[cont_col].mean())
            df['{}_{}_Std'.format(time_col, cont_col)] = df[time_col].map(pd.concat([X_train[[cont_col, time_col]], X_test[[cont_col, time_col]]], ignore_index=True).groupby(time_col)[cont_col].std())
            
            gc.collect()
'''

with faith('8. Fixing UserAgent, id_31 etc and Adding cents col etc...') as f:
	for df in tqdm([X_train, X_test]):
	    # ParsingError
	    df['ParsingError'] = np.nan
	    df.loc[df.query('~DeviceInfo.isnull() or ~UserAgent.isnull()', engine='python').index, 'ParsingError'] = 0
	    df.loc[df.query('~UserAgent.isnull()', engine='python').index, 'ParsingError'] = 1
	    
	    # BrowserUpToDate
	    df['BrowserUpToDate'] = np.nan
	    df.loc[df.query('~id_31.isnull()', engine='python').index, 'BrowserUpToDate'] = 0
	    df.loc[df.query('id_31 == "safari 10.0" and TransactionDate < "2017-09-19 00:00:00"').index, 'BrowserUpToDate'] = 1
	    df.loc[df.query('id_31 == "safari 11.0" and TransactionDate < "2018-09-17 00:00:00"').index, 'BrowserUpToDate'] = 1
	    df.loc[df.query('id_31 == "safari 12.0"').index, 'BrowserUpToDate'] = 1
	    df.loc[df.query('id_31 == "chrome 62.0" and TransactionDate < "2017-12-05 00:00:00"').index, 'BrowserUpToDate'] = 1
	    df.loc[df.query('id_31 == "chrome 63.0" and TransactionDate < "2018-01-24 00:00:00"').index, 'BrowserUpToDate'] = 1
	    df.loc[df.query('id_31 == "chrome 64.0" and TransactionDate < "2018-03-06 00:00:00"').index, 'BrowserUpToDate'] = 1
	    df.loc[df.query('id_31 == "chrome 65.0" and TransactionDate < "2018-04-17 00:00:00"').index, 'BrowserUpToDate'] = 1
	    df.loc[df.query('id_31 == "chrome 66.0" and TransactionDate < "2018-05-29 00:00:00"').index, 'BrowserUpToDate'] = 1
	    df.loc[df.query('id_31 == "chrome 67.0" and TransactionDate < "2018-07-24 00:00:00"').index, 'BrowserUpToDate'] = 1
	    df.loc[df.query('id_31 == "chrome 68.0" and TransactionDate < "2018-09-04 00:00:00"').index, 'BrowserUpToDate'] = 1
	    df.loc[df.query('id_31 == "chrome 69.0" and TransactionDate < "2018-10-16 00:00:00"').index, 'BrowserUpToDate'] = 1
	    df.loc[df.query('id_31 == "chrome 70.0" and TransactionDate < "2018-12-04 00:00:00"').index, 'BrowserUpToDate'] = 1
	    df.loc[df.query('id_31 == "chrome 71.0"').index, 'BrowserUpToDate'] = 1
	    df.loc[df.query('id_31 == "edge 16.0" and TransactionDate < "2018-04-30 00:00:00"').index, 'BrowserUpToDate'] = 1
	    df.loc[df.query('id_31 == "edge 17.0" and TransactionDate < "2018-11-13 00:00:00"').index, 'BrowserUpToDate'] = 1
	    df.loc[df.query('id_31 == "edge 18.0"').index, 'BrowserUpToDate'] = 1
	    df.loc[df.query('id_31 == "opera 49.0" and TransactionDate < "2018-01-04 00:00:00"').index, 'BrowserUpToDate'] = 1
	    df.loc[df.query('id_31 == "opera 51.0" and TransactionDate < "2018-03-22 00:00:00"').index, 'BrowserUpToDate'] = 1
	    df.loc[df.query('id_31 == "opera 52.0" and TransactionDate < "2018-05-10 00:00:00"').index, 'BrowserUpToDate'] = 1
	    df.loc[df.query('id_31 == "opera 53.0" and TransactionDate < "2018-06-28 00:00:00"').index, 'BrowserUpToDate'] = 1
	    df.loc[df.query('id_31 == "opera 54.0" and TransactionDate < "2018-08-16 00:00:00"').index, 'BrowserUpToDate'] = 1
	    df.loc[df.query('id_31 == "opera 55.0" and TransactionDate < "2018-09-25 00:00:00"').index, 'BrowserUpToDate'] = 1
	    df.loc[df.query('id_31 == "opera 56.0" and TransactionDate < "2018-11-28 00:00:00"').index, 'BrowserUpToDate'] = 1
	    df.loc[df.query('id_31 == "samsung browser 6.2" and TransactionDate < "2018-02-19 00:00:00"').index, 'BrowserUpToDate'] = 1
	    df.loc[df.query('id_31 == "samsung browser 6.4" and TransactionDate < "2018-06-07 00:00:00"').index, 'BrowserUpToDate'] = 1
	    df.loc[df.query('id_31 == "samsung browser 7.0" and TransactionDate < "2018-07-07 00:00:00"').index, 'BrowserUpToDate'] = 1
	    df.loc[df.query('id_31 == "samsung browser 7.2" and TransactionDate < "2018-08-19 00:00:00"').index, 'BrowserUpToDate'] = 1
	    df.loc[df.query('id_31 == "samsung browser 7.4" and TransactionDate < "2018-12-21 00:00:00"').index, 'BrowserUpToDate'] = 1
	    df.loc[df.query('id_31 == "samsung browser 8.2"').index, 'BrowserUpToDate'] = 1
	    df.loc[df.query('id_31 == "firefox 56.0" and TransactionDate < "2017-11-14 00:00:00"').index, 'BrowserUpToDate'] = 1
	    df.loc[df.query('id_31 == "firefox 57.0" and TransactionDate < "2018-01-23 00:00:00"').index, 'BrowserUpToDate'] = 1
	    df.loc[df.query('id_31 == "firefox 58.0" and TransactionDate < "2018-03-13 00:00:00"').index, 'BrowserUpToDate'] = 1
	    df.loc[df.query('id_31 == "firefox 59.0" and TransactionDate < "2018-05-09 00:00:00"').index, 'BrowserUpToDate'] = 1
	    df.loc[df.query('id_31 == "firefox 60.0" and TransactionDate < "2018-06-26 00:00:00"').index, 'BrowserUpToDate'] = 1
	    df.loc[df.query('id_31 == "firefox 61.0" and TransactionDate < "2018-09-05 00:00:00"').index, 'BrowserUpToDate'] = 1
	    df.loc[df.query('id_31 == "firefox 62.0" and TransactionDate < "2018-10-23 00:00:00"').index, 'BrowserUpToDate'] = 1
	    df.loc[df.query('id_31 == "firefox 63.0" and TransactionDate < "2018-12-11 00:00:00"').index, 'BrowserUpToDate'] = 1
	    df.loc[df.query('id_31 == "firefox 64.0"').index, 'BrowserUpToDate'] = 1
	    df.loc[df.query('id_31 == "firefox 65.0"').index, 'BrowserUpToDate'] = 1
	    
	    # TransactionAmtCents
	    df['TransactionAmtCents'] = df['TransactionAmt'] - df['TransactionAmt'].astype('int')

with faith('9. Label + Frequency Encoding M and various ID cols') as f:
	for df in tqdm([X_train, X_test]):
	    # Rounding
	    df['id_11'] = df['id_11'].round(2)
	    
	    # Casting
	    df['id_33'] = df['id_33'].str.split('x', expand=True)[0].astype(np.float32) * df['id_33'].str.split('x', expand=True)[1].astype(np.float32)
	    
	    # Label Encoding
	    df['M1'] = df['M1'].map({'F': 0, 'T': 1})
	    df['M2'] = df['M2'].map({'F': 0, 'T': 1})
	    df['M3'] = df['M3'].map({'F': 0, 'T': 1})
	    df['M4'] = df['M4'].map({'M0': 0, 'M1': 1, 'M2': 2})
	    df['M5'] = df['M5'].map({'F': 0, 'T': 1})
	    df['M6'] = df['M6'].map({'F': 0, 'T': 1})
	    df['M7'] = df['M7'].map({'F': 0, 'T': 1})
	    df['M8'] = df['M8'].map({'F': 0, 'T': 1})
	    df['M9'] = df['M9'].map({'F': 0, 'T': 1})
	    df['id_12'] = df['id_12'].map({'NotFound': 0, 'Found': 1})
	    df['id_15'] = df['id_15'].map({'Unknown': 0, 'New': 1, 'Found': 2})
	    df['id_16'] = df['id_16'].map({'NotFound': 0, 'Found': 1})
	    df['id_23'] = df['id_23'].map({'IP_PROXY:TRANSPARENT': 0, 'IP_PROXY:ANONYMOUS': 1, 'IP_PROXY:HIDDEN': 2})
	    df['id_27'] = df['id_27'].map({'Found': 1})
	    df['id_28'] = df['id_28'].map({'New': 0, 'Found': 1})
	    df['id_29'] = df['id_29'].map({'NotFound': 0, 'Found': 1})
	    df['id_34'] = df['id_34'].map({'match_status:-1': -1, 'match_status:0': 0, 'match_status:1': 1, 'match_status:2': 2})
	    df['id_35'] = df['id_35'].map({'F': 0, 'T': 1})
	    df['id_36'] = df['id_36'].map({'F': 0, 'T': 1})
	    df['id_37'] = df['id_37'].map({'F': 0, 'T': 1})
	    df['id_38'] = df['id_38'].map({'F': 0, 'T': 1})
	    
	    # Frequency Encoding    
	    for col in CAT_COLS + ['TransactionAmt', 'TransactionAmtCents']:
	        df[f'{col}_VC'] = df[col].map(pd.concat([X_train[col], X_test[col]], ignore_index=True).value_counts(dropna=False))

def build_ranges(ranges):
    out = []
    for arange in ranges:
        out.append(np.arange(arange[0], arange[-1]+1, 1).tolist())
    return sum(out, [])

# from SO [link TO-DO]
def sliding_window(iterable, size=2, step=1, fillvalue=None):
    if size < 0 or step < 1:
        raise ValueError
        
    it = iter(iterable)
    q = deque(islice(it, size), maxlen=size)
    
    if not q:
        return  # empty iterable or size == 0
    q.extend(fillvalue for _ in range(size - len(q)))  # pad to size
    
    while True:
        yield iter(q)  # iter() to avoid accidental outside modifications #can be changed 
        # as per our req we don't need an iter, we need the indices
        try:
            q.append(next(it))
        except StopIteration: # Python 3.5 pep 479 support
            return
        q.extend(next(it, fillvalue) for _ in range(step - 1))

with faith('10. mapping domain email cols etc') as f:
	for data in tqdm([X_train, X_test]):
		data['pemail_domain']  = data.P_emaildomain.astype(str).apply(domain)
		data['pemail_ext']     = data.P_emaildomain.astype(str).apply(pemail_country).map(country_map)
		data['remail_domain']  = data.R_emaildomain.astype(str).apply(domain)
		data['remail_ext']     = data.R_emaildomain.astype(str).apply(pemail_country).map(country_map)
		data['p_and_r_email']  = data.P_emaildomain.astype(str) + ' ' + data.R_emaildomain.astype(str)

cont_cols = ['TransactionAmt','dist1']
# mem intensive
def calculate_rolling_feats(df, periods=periods, min_instances=min_instances, aggr_cols=aggr_cols, cont_cols=cont_cols):

    for period in periods:
        for col in tqdm(aggr_cols):
            # For aggregate values appearing 1000x or more:
            vcs = df[col].value_counts()
            vcs = vcs[vcs>min_instances].index.values
            mask = ~df[col].isin(vcs)

            #For these two continuous columns:
            # TODO: Experiment w/ other high card cont columns V* here such as: 'V307', 'V314'??
            # Chosen for having low nans and high roc
            for cont in cont_cols:
                # Calculate rolling period mean and mean_diffs:
                new_colA = '{}_mean__{}_group_{}'.format(period, cont, col)
                new_colB = cont + '_-_' + new_colA
                
                temp = df.groupby(col + ['ProductCD']).rolling(period, on='TransactionDate')[cont].mean().reset_index()
                temp.rename(columns={cont:new_colA}, inplace=True)
                temp.drop_duplicates(['TransactionDate', col, 'ProductCD'], inplace=True)
                df = df.merge(temp, how='left', on=['TransactionDate', col, 'ProductCD'])
                df[new_colB] = df[cont] - df[new_colA]

                # NAN out any newly generated col where our groupby col,
                # the aggregate, appears less than 1000x in the dset:
                df.loc[mask, new_colA] = np.nan
                df.loc[mask, new_colB] = np.nan
    return df

with faith('11. Adding Rolling mean feats') as f:
	X_train = calculate_rolling_feats(X_train, periods, min_instances, aggr_cols, cont_cols)
	X_test = calculate_rolling_feats(X_test, periods, min_instances, aggr_cols, cont_cols)

# Count of M1=T transactions having the sample in question's addr in the past week
# Count of M1=T transactions having the sample in question's card1-6 in the past week
# Count of M1=T transactions having the sample in question's productcd in the past week
# Count of M1=T transactions having the sample in question's r/e email domain in the past week
CARD_COLS=[f'card{i}' for i in range(1,7)]

with faith('12. Adding m1==1(t) feats aggs etc') as f:
	for df in tqdm([X_train, X_test]):
	    for col in CARD_COLS + ['addr1', 'ProductCD', 'P_emaildomain', 'R_emaildomain']:
	        df['M1T'] = df.M1== 1
	        temp = df.groupby(f'{col}').rolling('7d', min_periods=7, on='TransactionDate').M1T.sum().reset_index()
	        temp.rename(columns={'M1T':'M1T_7D_{col}'}, inplace=True)
	        temp.drop_duplicates([f'{col}', 'TransactionDate'], inplace=True)
	        del df['M1T']
	        df = df.merge(temp, how='left', on=['TransactionDate', f'{col}'])
	        del temp
	        gc.collect()

# DOWNCASTING INT COLS
INT64_COLS = [col for col in X_test.columns if X_test[col].dtype == 'int64']
with faith('13. Downcasting feats....') as f:
	for df in [X_train, X_test]:
	    for col in tqdm(INT64_COLS):
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

with faith('14. Caching files etc and otehr new cols') as f:
	X_train.to_csv('__preprocessed_train.csv', index=None)
	X_test.to_csv('__preprocessed_test.csv', index=None)
	xtra_cols_added = list(set(X_train.columns) - set(org_cols))
	pd.Series(xtra_cols_added).to_hdf('new_cols_added_in_preprocess_and_fes.hdf', key='preprocess')
	pd.Series(org_cols).to_hdf('org_cols_raw_data.hdf', key='raw')

print('DONE')

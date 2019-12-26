# %% [code]
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))

#https://www.kaggle.com/bminixhofer/xlearn 
#Thanks To Benjamin Minixhofer ...

os.environ['USER'] = 'root'
os.system('pip install ../input/xlearn/xlearn/xlearn-0.40a1/')

import xlearn as xl

# %% [code] {"scrolled":true}
dtypes = {
    'Type' :'int32',
    'Name' : 'category',
    'Age' : 'float16',
    'Breed1' : 'category',
    'Breed2' : 'category',
    'Gender' : 'category',
    'Color1':'category',
    'Color2' : 'category',
    'Color3' : 'category',
    'MaturitySize' : 'category',
    'FurLength':'category',
    'Vaccinated':'category',
    'Dewormed':'category',
    'Sterilized':'category',
    'Health':'category',
    'Quantity':'int16',
    'Fee':'int32',
    'State':'category',
    'RescuerID' : 'category',
    'VideoAmt':'int16',
    'PhotoAmt': 'int16','Description':'object','PetID':'category','AdoptionSpeed':'int'
}

# %% [code]
from tqdm import tqdm_notebook as tqdm

# %% [code]
import os
from collections import defaultdict
from csv import DictReader
import math

train_path = '../input/petfinder-adoption-prediction/train/train.csv'
dont_use = ['RescuerID', 'AdoptionSpeed', 'Description', 'PetID']
num_cols = ['Age', 'Fee', 'PhotoAmt', 'VideoAmt']
too_many_vals = [""]

categories = [k for k, v in dtypes.items() if k not in dont_use]
categories_index = dict(zip(categories, range(len(categories))))
print(categories)
field_features = defaultdict()

# %% [code]
#Abhisekhs' Malware Discussion https://www.kaggle.com/c/microsoft-malware-prediction/discussion/75217#latest-457846
max_val = 1
with open('train.libffm', 'a') as the_file:
    for t, row in tqdm(enumerate(DictReader(open(train_path)))):
        if t % 1000 == 0:
            print(t, len(field_features), max_val)
        label = [row['AdoptionSpeed']]
        ffeatures = []

        for field in categories:
            if field == 'AdoptionSpeed':
                continue
            feature = row[field]
            if feature == '':
                feature = "unk"
            if field not in num_cols:
                ff = field + '_____' + feature
            else:
                if feature == "unk" or float(feature) == -1:
                    ff = field + '_____' + str(0)
                else:
                    if field in too_many_vals:
                        ff = field + '_____' + str(int(round(math.log(1 + float(feature)))))
                    else:
                        ff = field + '_____' + str(int(round(float(feature))))
            if ff not in field_features:
                if len(field_features) == 0:
                    field_features[ff] = 1
                    max_val += 1
                else:
                    field_features[ff] = max_val + 1
                    max_val += 1

            fnum = field_features[ff]
            ffeatures.append('{}:{}:1'.format(categories_index[field], fnum))
            
        line = label + ffeatures
        the_file.write('{}\n'.format(' '.join(line)))

# %% [code]
##Abhisekhs' Malware Discussion https://www.kaggle.com/c/microsoft-malware-prediction/discussion/75217#latest-457846

test_path = '../input/petfinder-adoption-prediction/test/test.csv'
with open('test.libffm', 'a') as the_file:
    for t, row in tqdm(enumerate(DictReader(open(test_path)))):
        if t % 1000 == 0:
            print(t, len(field_features), max_val)
        #label = [row['AdoptionSpeed']]
        label = [str(0)]
        ffeatures = []

        for field in categories:
            if field == 'AdoptionSpeed':
                continue
            feature = row[field]
            if feature == '':
                feature = "unk"
            if field not in num_cols:
                ff = field + '_____' + feature
            else:
                if feature == "unk" or float(feature) == -1:
                    ff = field + '_____' + str(0)
                else:
                    if field in too_many_vals:
                        ff = field + '_____' + str(int(round(math.log(1 + float(feature)))))
                    else:
                        ff = field + '_____' + str(int(round(float(feature))))
            if ff not in field_features:
                if len(field_features) == 0:
                    field_features[ff] = 1
                    max_val += 1
                else:
                    field_features[ff] = max_val + 1
                    max_val += 1

            fnum = field_features[ff]

            ffeatures.append('{}:{}:1'.format(categories_index[field], fnum))
        line = label + ffeatures
        the_file.write('{}\n'.format(' '.join(line)))

# %% [code]
import xlearn as xl

# create ffm model
ffm_model = xl.create_ffm() 

# set training
ffm_model.setTrain("train.libffm")

# %% [code]
%%time
# define params
param = {'task':'reg', 'lr':0.2,
         'lambda':0.002, 'metric':'rmse', 'epoch' : 50}

# train the model
ffm_model.fit(param, 'xl.out')

# %% [code]
# set the test data
ffm_model.setTest("test.libffm")

# make predictions
ffm_model.predict("xl.out", "output_new.txt")

# %% [code]
# create submission file
from matplotlib import pyplot as plt
sample = pd.read_csv('../input/petfinder-adoption-prediction/test/sample_submission.csv')
output = pd.read_csv('output_new.txt', header=None)[0].values

plt.hist(output);

# %% [code]
def predict(X, coef):
    X_p = np.copy(X)
    for i, pred in enumerate(X_p):
        if pred < coef[0]:
            X_p[i] = 0
        elif pred >= coef[0] and pred < coef[1]:
            X_p[i] = 1
        elif pred >= coef[1] and pred < coef[2]:
            X_p[i] = 2
        elif pred >= coef[2] and pred < coef[3]:
            X_p[i] = 3
        else:
            X_p[i] = 4
    return X_p

# %% [code]
Coefficients =  [1.645, 2.115, 2.51, 2.845]
output = predict(output, Coefficients)

# %% [code]
sample.AdoptionSpeed = output
sample['AdoptionSpeed'] = sample['AdoptionSpeed'].astype(int)
sample.to_csv('submission.csv', index=False)

# %% [code]
sample.head()

# %% [code]
from collections import Counter
Counter(output)

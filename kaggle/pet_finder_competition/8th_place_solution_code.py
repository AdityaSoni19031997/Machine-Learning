import cv2
import os
# os.environ['USER'] = 'root'
# os.system('pip install ../input/xlearn/xlearn/xlearn-0.40a1/')
# import xlearn as xl
from collections import defaultdict
from csv import DictReader
import math
import warnings
warnings.filterwarnings("ignore")

import time
import gc
import glob
import ujson as json
import pprint
import joblib
import warnings
import random

import pandas as pd
import numpy as np
import seaborn as sns
import scipy as sp
import matplotlib.pyplot as plt
import lightgbm as lgb
import xgboost as xgb
import tensorflow as tf

from collections import Counter
from functools import partial
from math import sqrt
from sklearn.metrics import cohen_kappa_score, mean_squared_error, accuracy_score
from sklearn.metrics import confusion_matrix as sk_cmatrix
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import SparsePCA, TruncatedSVD, LatentDirichletAllocation, NMF

from PIL import Image as PILImage
from joblib import Parallel, delayed
from tqdm import tqdm
from contextlib import contextmanager
from pandas.io.json import json_normalize

from keras.preprocessing.text import Tokenizer
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Input, Lambda, AveragePooling1D, CuDNNLSTM, CuDNNGRU
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, CuDNNLSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D
from keras.layers import Bidirectional, GlobalMaxPool1D, GlobalMaxPooling2D, GlobalAveragePooling1D, GlobalAvgPool2D, GlobalMaxPool2D
from keras.layers import Input, Embedding, Dense, Conv2D, MaxPool2D, concatenate
from keras.layers import Reshape, Flatten, Concatenate, Dropout, SpatialDropout1D, SpatialDropout2D
from keras.applications.densenet import preprocess_input, DenseNet121
from keras.optimizers import Adam
from keras.models import Model
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints, optimizers, layers

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

import psutil
from multiprocessing import Pool

import fastai
from fastai.vision import *
print('fast.ai version:{}'.format(fastai.__version__))

num_partitions = 20  # number of partitions to split dataframe
num_cores = psutil.cpu_count()  # number of cores on your machine

print('number of cores:', num_cores)

def df_parallelize_run(df, func):
    
    df_split = np.array_split(df, num_partitions)
    pool = Pool(num_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    
    return df

# From Quora kaggle Comp's (latest one)
import re
# remove space
spaces = ['\u200b', '\u200e', '\u202a', '\u202c', '\ufeff', '\uf0d8', '\u2061', '\x10', '\x7f', '\x9d', '\xad', '\xa0']
def remove_space(text):
    """
    remove extra spaces and ending space if any
    """
    for space in spaces:
        text = text.replace(space, ' ')
    text = text.strip()
    text = re.sub('\s+', ' ', text)
    return text

# replace strange punctuations and raplace diacritics
from unicodedata import category, name, normalize

def remove_diacritics(s):
    return ''.join(c for c in normalize('NFKD', s.replace('ø', 'o').replace('Ø', 'O').replace('⁻', '-').replace('₋', '-'))
                  if category(c) != 'Mn')

special_punc_mappings = {
    "—": "-", "–": "-", "_": "-", '”': '"', "″": '"', '“': '"', '•': '.', '−': '-',
    "’": "'", "‘": "'", "´": "'", "`": "'", '\u200b': ' ', '\xa0': ' ','،':'','„':'',
    '…': ' ... ', '\ufeff': ''
}

def clean_special_punctuations(text):
    for punc in special_punc_mappings:
        if punc in text:
            text = text.replace(punc, special_punc_mappings[punc])
    text = remove_diacritics(text)
    return text

# clean numbers
def clean_number(text):
    text = re.sub(r'(\d+)([a-zA-Z])', '\g<1> \g<2>', text)
    text = re.sub(r'(\d+) (th|st|nd|rd) ', '\g<1>\g<2> ', text)
    text = re.sub(r'(\d+),(\d+)', '\g<1>\g<2>', text)
    return text

import string
regular_punct = list(string.punctuation)
extra_punct = [
    ',', '.', '"', ':', ')', '(', '!', '?', '|', ';', "'", '$', '&',
    '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£',
    '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',
    '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', '“', '★', '”',
    '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾',
    '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', '▒', '：', '¼', '⊕', '▼',
    '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲',
    'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', '∙', '）', '↓', '、', '│', '（', '»',
    '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø',
    '¹', '≤', '‡', '√', '«', '»', '´', 'º', '¾', '¡', '§', '£', '₤',
    ':)', ': )', ':-)', '(:', '( :', '(-:', ':\')',
    ':D', ': D', ':-D', 'xD', 'x-D', 'XD', 'X-D',
    '<3', ':*',
    ';-)', ';)', ';-D', ';D', '(;',  '(-;',
    ':-(', ': (', ':(', '\'):', ')-:',
    '-- :','(', ':\'(', ':"(\'',]

def handle_emojis(text):
    # Smile -- :), : ), :-), (:, ( :, (-:, :')
    text = re.sub(r'(:\s?\)|:-\)|\(\s?:|\(-:|:\'\))', ' EMO_POS ', text)
    # Laugh -- :D, : D, :-D, xD, x-D, XD, X-D
    text = re.sub(r'(:\s?D|:-D|x-?D|X-?D)', ' EMO_POS ', text)
    # Love -- <3, :*
    text = re.sub(r'(<3|:\*)', ' EMO_POS ', text)
    # Wink -- ;-), ;), ;-D, ;D, (;,  (-;
    text = re.sub(r'(;-?\)|;-?D|\(-?;)', ' EMO_POS ', text)
    # Sad -- :-(, : (, :(, ):, )-:
    text = re.sub(r'(:\s?\(|:-\(|\)\s?:|\)-:)', ' EMO_NEG ', text)
    # Cry -- :,(, :'(, :"(
    text = re.sub(r'(:,\(|:\'\(|:"\()', ' EMO_NEG ', text)
    return text

def stop(text):
    
    from nltk.corpus import stopwords
    
    text = " ".join([w.lower() for w in text.split()])
    stop_words = stopwords.words('english')
    
    words = [w for w in text.split() if not w in stop_words]
    return " ".join(words)

all_punct = list(set(regular_punct + extra_punct))
# do not spacing - and .
all_punct.remove('-')
all_punct.remove('.')

# clean repeated letters
def clean_repeat_words(text):
    
    text = text.replace("img", "ing")
    text = re.sub(r"(I|i)(I|i)+ng", "ing", text)
    text = re.sub(r"(L|l)(L|l)(L|l)+y", "lly", text)
    text = re.sub(r"(A|a)(A|a)(A|a)+", "a", text)
    text = re.sub(r"(C|c)(C|c)(C|c)+", "cc", text)
    text = re.sub(r"(D|d)(D|d)(D|d)+", "dd", text)
    text = re.sub(r"(E|e)(E|e)(E|e)+", "ee", text)
    text = re.sub(r"(F|f)(F|f)(F|f)+", "ff", text)
    text = re.sub(r"(G|g)(G|g)(G|g)+", "gg", text)
    text = re.sub(r"(I|i)(I|i)(I|i)+", "i", text)
    text = re.sub(r"(K|k)(K|k)(K|k)+", "k", text)
    text = re.sub(r"(L|l)(L|l)(L|l)+", "ll", text)
    text = re.sub(r"(M|m)(M|m)(M|m)+", "mm", text)
    text = re.sub(r"(N|n)(N|n)(N|n)+", "nn", text)
    text = re.sub(r"(O|o)(O|o)(O|o)+", "oo", text)
    text = re.sub(r"(P|p)(P|p)(P|p)+", "pp", text)
    text = re.sub(r"(Q|q)(Q|q)+", "q", text)
    text = re.sub(r"(R|r)(R|r)(R|r)+", "rr", text)
    text = re.sub(r"(S|s)(S|s)(S|s)+", "ss", text)
    text = re.sub(r"(T|t)(T|t)(T|t)+", "tt", text)
    text = re.sub(r"(V|v)(V|v)+", "v", text)
    text = re.sub(r"(Y|y)(Y|y)(Y|y)+", "y", text)
    text = re.sub(r"plzz+", "please", text)
    text = re.sub(r"(Z|z)(Z|z)(Z|z)+", "zz", text)
    return text

def spacing_punctuation(text):
    """
    add space before and after punctuation and symbols
    """
    for punc in all_punct:
        if punc in text:
            text = text.replace(punc, f' {punc} ')
    return text

def preprocess(text):
    """
    preprocess text main steps
    """
    text = remove_space(text)
    text = clean_special_punctuations(text)
    text = handle_emojis(text)
    text = clean_number(text)
    text = spacing_punctuation(text)
    text = clean_repeat_words(text)
    text = stop(text)
    return text
####################################################################
#cat cats dog dogs adoption adopted
preload = False

class PetFinderParser(object):

    def __init__(self, debug=False):

        self.debug = debug
        self.sentence_sep = ' '

        # Does not have to be extracted because main DF already contains description
        self.extract_sentiment_text = False

    def open_metadata_file(self, filename):
        """
        Load metadata file.
        """
        with open(filename, 'r') as f:
            metadata_file = json.load(f)
        return metadata_file

    def open_sentiment_file(self, filename):
        """
        Load sentiment file.
        """
        with open(filename, 'r') as f:
            sentiment_file = json.load(f)
        return sentiment_file

    def open_image_file(self, filename):
        """
        Load image file.
        """
        image = np.asarray(PILImage.open(filename))
        return image

    def parse_sentiment_file(self, file):
        """
        Parse sentiment file. Output DF with sentiment features.
        """

        file_sentiment = file['documentSentiment']
        file_entities = [x['name'] for x in file['entities']]
        file_entities = self.sentence_sep.join(file_entities)
        
        try:
            file_salience_mean = np.asarray([x['salience'] for x in file['entities']]).mean()
            file_salience_max  = np.asarray([x['salience'] for x in file['entities']]).max()
            file_salience_min  = np.asarray([x['salience'] for x in file['entities']]).min()
            file_salience_len  = len(np.asarray([x['salience'] for x in file['entities']]))
        except:
            file_salience_mean, file_salience_max, file_salience_min, file_salience_len = np.nan, np.nan, np.nan, np.nan
        
        try:
            file_mentions = np.asarray([len(x['mentions']) for x in file['entities']]).mean()
            file_mentions_total = np.asarray([len(x['mentions']) for x in file['entities']]).sum()
            file_mentions_len = len(np.asarray([len(x['mentions']) for x in file['entities']]))
        except:
            file_mentions, file_mentions_len, file_mentions_total = np.nan, np.nan, np.nan
        
        file_entities_new =[x['name'] for x in file['entities']]
        file_entities_new = len(self.sentence_sep.join(file_entities_new))

        if self.extract_sentiment_text:
            file_sentences_text = [x['text']['content'] for x in file['sentences']]
            file_sentences_text = self.sentence_sep.join(file_sentences_text)
        file_sentences_sentiment = [x['sentiment'] for x in file['sentences']]

        file_sentences_sentiment = pd.DataFrame.from_dict(file_sentences_sentiment, orient='columns').sum()
        file_sentences_sentiment = file_sentences_sentiment.add_prefix('document_').to_dict()

        file_sentiment.update(file_sentences_sentiment)

        df_sentiment = pd.DataFrame.from_dict(file_sentiment, orient='index').T
        if self.extract_sentiment_text:
            df_sentiment['text'] = file_sentences_text

        df_sentiment['entities'] = file_entities
        df_sentiment['entities_name'] = file_entities_new
        df_sentiment['salience_mean'] = file_salience_mean
        df_sentiment['salience_max']  = file_salience_max
        df_sentiment['salience_min']  = file_salience_min
        df_sentiment['salience_len']  = file_salience_len
        df_sentiment['mentions_mean_length'] = file_mentions
        df_sentiment['mentions_len'] = file_mentions_len
        df_sentiment['mentions_total'] = file_mentions_total
        
        df_sentiment = df_sentiment.add_prefix('sentiment_')

        return df_sentiment
    
    #updated this in latest run
    def parse_metadata_file(self, file):
        """
        Parse metadata file. Output DF with metadata features.
        """
        
        file_keys = list(file.keys())
        
        if 'faceAnnotations' in file_keys:
            
            file_annots_text = file['faceAnnotations'][:int(len(file['faceAnnotations']))]
            rollAngle = np.asarray([x['rollAngle'] for x in file_annots_text]).mean()
            panAngle = np.asarray([x['panAngle'] for x in file_annots_text]).mean()
            tiltAngle = np.asarray([x['tiltAngle'] for x in file_annots_text]).mean()
            detectionConfidence = np.asarray([x['detectionConfidence'] for x in file_annots_text]).mean()
            landmarkingConfidence = np.asarray([x['landmarkingConfidence'] for x in file_annots_text]).mean()
            # joy_likelihood = np.asarray(x['joyLikelihood'] for x in file_annots_text])
            # sorrowLikelihood = np.asarray(x['sorrowLikelihood'] for x in file_annots_text])
            # underExposedLikelihood = np.asarray(x['underExposedLikelihood'] for x in file_annots_text])
            # blurredLikelihood = np.asarray(x['blurredLikelihood'] for x in file_annots_text])
            
            try:
                top_left_x_face = np.asarray([float(x["boundingPoly"]["vertices"][0]['x']) for x in file_annots_text]).mean()
            except:
                top_left_x_face = np.asarray([0])
            try:
                top_left_y_face = np.asarray([float(x["boundingPoly"]["vertices"][0]['y']) for x in file_annots_text]).mean()
            except:
                top_left_y_face = np.asarray([0])
            try:
                bottom_right_x_face = np.asarray([float(x["boundingPoly"]["vertices"][2]['x']) for x in file_annots_text]).mean()
            except:
                bottom_right_x_face = np.asarray(0)
            try:
                bottom_right_y_face = np.asarray([float(x["boundingPoly"]["vertices"][2]['y']) for x in file_annots_text]).mean()
            except:
                bottom_right_y_face = np.asarray(0)
    
            try:
                top_left_x_fd_face = np.asarray([float(x["fdBoundingPoly"]["vertices"][0]['x']) for x in file_annots_text]).mean()
            except:
                top_left_x_fd_face = np.asarray(0)
            try:
                top_left_y_fd_face = np.asarray([float(x["fdBoundingPoly"]["vertices"][0]['y']) for x in file_annots_text]).mean()
            except:
                top_left_y_fd_face = np.asarray(0)
            try:
                bottom_right_x_fd_face = np.asarray([float(x["fdBoundingPoly"]["vertices"][2]['x']) for x in file_annots_text]).mean()
            except:
                bottom_right_x_fd_face = np.asarray(0)
            try:
                bottom_right_y_fd_face = np.asarray([float(x["fdBoundingPoly"]["vertices"][2]['y']) for x in file_annots_text]).mean()
            except:
                bottom_right_y_fd_face = np.asarray(0)
        else:
            rollAngle, panAngle, tiltAngle, detectionConfidence, landmarkingConfidence = np.asarray(0),np.asarray(0),np.asarray(0),np.asarray(0),np.asarray(0)
            top_left_x_fd_face, top_left_x_face = np.asarray(0), np.asarray(0)
            top_left_y_fd_face , top_left_y_face= np.asarray(0), np.asarray(0)
            bottom_right_x_fd_face, bottom_right_x_face= np.asarray(0), np.asarray(0)
            bottom_right_y_fd_face, bottom_right_y_face = np.asarray(0), np.asarray(0)
        
        if 'textAnnotations' in file_keys:
            
            file_annots_text = file['textAnnotations'][:int(len(file['textAnnotations']))]
            file_top_desc_text_annot = np.asarray([x['description'] for x in file_annots_text])

            locale = np.asarray([x['locale'] for x in file_annots_text])
            #poly's
            try:
                top_left_x = np.asarray([x["boundingPoly"]["vertices"][0]['x'] for x in file_annots_text])
            except:
                top_left_x = np.asarray(0)
            try:
                top_left_y = np.asarray([x["boundingPoly"]["vertices"][0]['y'] for x in file_annots_text])
            except:
                top_left_y = np.asarray(0)
            try:
                bottom_right_x = np.asarray([x["boundingPoly"]["vertices"][2]['x'] for x in file_annots_text])
            except:
                bottom_right_x = np.asarray(0)
            try:
                bottom_right_y =np.asarray([x["boundingPoly"]["vertices"][2]['y'] for x in file_annots_text])
            except:
                bottom_right_y = np.asarray(0)
        else:
            file_top_desc_text_annot = ['']
            locale = np.asarray('en')
            top_left_x = np.asarray(0)
            top_left_y = np.asarray(0)
            bottom_right_x = np.asarray(0)
            bottom_right_y = np.asarray(0)
            
        if 'labelAnnotations' in file_keys:
            file_annots = file['labelAnnotations'][:int(len(file['labelAnnotations']))]
            file_top_score = np.asarray([x['score'] for x in file_annots]).mean()
            file_top_desc = [x['description'] for x in file_annots]
        else:
            file_top_score = np.nan
            file_top_desc = ['']
        
        file_colors = file['imagePropertiesAnnotation']['dominantColors']['colors']
        file_crops = file['cropHintsAnnotation']['cropHints']
        file_color_count = np.asarray(len(file_colors))

        file_color_score = np.asarray([x['score'] for x in file_colors]).mean()
        file_color_pixelfrac = np.asarray([x['pixelFraction'] for x in file_colors]).mean()

        file_crop_conf = np.asarray([x['confidence'] for x in file_crops]).mean()
        
        if 'importanceFraction' in file_crops[0].keys():
            file_crop_importance = np.asarray([x['importanceFraction'] for x in file_crops]).mean()
        else:
            file_crop_importance = np.nan

        df_metadata = {
            'color_count': file_color_count,
            'annots_score': file_top_score,
            'color_score': file_color_score,
            'color_pixelfrac': file_color_pixelfrac,
            'crop_conf': file_crop_conf,
            'crop_importance': file_crop_importance,
            'annots_top_desc': self.sentence_sep.join(file_top_desc),
            ##### text ones added which we missed completely
            'text_annots_desc': self.sentence_sep.join(file_top_desc_text_annot),
            # 'locale': locale,
            'top_left_x_text': top_left_x,
            'top_left_y_text': top_left_y,
            'bottom_right_x_text': bottom_right_x,
            'bottom_right_y_text': bottom_right_y,
            
            'top_left_x_face': top_left_x_face,
            'top_left_y_face': top_left_y_face,
            'bottom_right_x_face': bottom_right_x_face,
            'bottom_right_y_face': bottom_right_y_face,
            
            'top_left_x_fd_face': top_left_x_fd_face,
            'top_left_y_fd_face': top_left_y_fd_face,
            'bottom_right_x_fd_face': bottom_right_x_fd_face,
            'bottom_right_y_fd_face': bottom_right_y_fd_face,
            ###### FaceAnnotations
            'rollAngle': rollAngle,
            'panAngle': panAngle,
            'tiltAngle': tiltAngle,
            'detectionConfidence': detectionConfidence,
            'landmarkingConfidence': landmarkingConfidence,
        }
        
        df_metadata = pd.DataFrame.from_dict(df_metadata, orient='index').T
        df_metadata = df_metadata.add_prefix('metadata_')
        
        return df_metadata

def confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):
    """
    Returns the confusion matrix between rater's ratings
    """
    assert (len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(rater_a + rater_b)
    if max_rating is None:
        max_rating = max(rater_a + rater_b)
    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(num_ratings)]
                for j in range(num_ratings)]
    for a, b in zip(rater_a, rater_b):
        conf_mat[a - min_rating][b - min_rating] += 1
    return conf_mat

def histogram(ratings, min_rating=None, max_rating=None):
    """
    Returns the counts of each type of rating that a rater made
    """
    if min_rating is None:
        min_rating = min(ratings)
    if max_rating is None:
        max_rating = max(ratings)
    num_ratings = int(max_rating - min_rating + 1)
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings:
        hist_ratings[r - min_rating] += 1
    return hist_ratings

def quadratic_weighted_kappa(y, y_pred):
    rater_a = y
    rater_b = y_pred
    min_rating = None
    max_rating = None
    rater_a = np.array(rater_a, dtype=int)
    rater_b = np.array(rater_b, dtype=int)
    assert (len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None:
        max_rating = max(max(rater_a), max(rater_b))
    conf_mat = confusion_matrix(rater_a, rater_b,
                                min_rating, max_rating)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j]
                              / num_scored_items)
            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    return (1.0 - numerator / denominator)

class OptimizedRounder(object):
    def __init__(self):
        self.coef_ = 0

    def _kappa_loss(self, coef, X, y):
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

        ll = quadratic_weighted_kappa(y, X_p)
        return -ll

    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = [0.5, 1.5, 2.5, 3.5]
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')

    def predict(self, X, coef):
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

    def coefficients(self):
        return self.coef_['x']

def rmse(actual, predicted):
    return sqrt(mean_squared_error(actual, predicted))

### Helper function for parallel data processing: Modified Slightly fixing few names and all 
#works Super fast Thanks to the Public Kernel !!!!!!!!!!
def extract_additional_features(pet_id, mode='train'):
    
    pet_parser = PetFinderParser()
    sentiment_filename = f'../input/petfinder-adoption-prediction/{mode}_sentiment/{pet_id}.json'
    try:
        sentiment_file = pet_parser.open_sentiment_file(sentiment_filename)
        df_sentiment   = pet_parser.parse_sentiment_file(sentiment_file)
        df_sentiment['PetID'] = pet_id
    except FileNotFoundError:
        df_sentiment = []

    dfs_metadata = []
    for ind in range(1,200):
        metadata_filename = '../input/petfinder-adoption-prediction/{}_metadata/{}-{}.json'.format(mode, pet_id, ind)
        try:
            metadata_file = pet_parser.open_metadata_file(metadata_filename)
            df_metadata   = pet_parser.parse_metadata_file(metadata_file)
            df_metadata['PetID'] = pet_id
            dfs_metadata.append(df_metadata)
        except FileNotFoundError:
            break
    if dfs_metadata:
        dfs_metadata = pd.concat(dfs_metadata, ignore_index=True, sort=False)
    dfs = [df_sentiment, dfs_metadata]
    return dfs

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)

#from mlcrate
import time
from warnings import warn

# Function wrapper that warns is deprecated, and call the function anyway
def _deprecated(func, old_name, new_name):
    def new_func(*args, **kwargs):
        message = '{}() has been deprecated in favour of {}() and will be removed soon'.format(old_name, new_name)
        warn(message)
        return func(*args, **kwargs)
    return new_func

class Timer:
    """A class for tracking timestamps and time elapsed since events. Useful for profiling code.

    Usage:
    >>> t = Timer()
    >>> t.since() # Seconds since the timetracker was initialised
    >>> t.add('func') # Save the current timestamp as 'func'
    >>> t.since('func') # Seconds since 'func' was added
    >>> t['func'] # Get the absolute timestamp of 'func' for other uses
    """
    def __init__(self):
        self.times = {}
        self.add(0)

    def __getitem__(self, key):
        return self.times[key]

    def add(self, key):
        """Add the current time to the index with the specified key"""
        self.times[key] = time.time()

    def since(self, key=0):
        """Get the time elapsed in seconds since the specified key was added to the index"""
        return time.time() - self.times[key]

    def fsince(self, key=0, max_fields=3):
        """Get the time elapsed in seconds, nicely formatted by format_duration()"""
        return format_duration(self.since(key), max_fields)

    elapsed = _deprecated(since, 'Timer.elapsed', 'Timer.since')
    format_elapsed = _deprecated(fsince, 'Timer.format_elapsed', 'Timer.fsince')

def now():
    """Returns the current time as a string in the format 'YYYY_MM_DD_HH_MM_SS'. Useful for timestamping filenames etc."""
    return time.strftime("%Y_%m_%d_%H_%M_%S")

# Alias for backwards-compatibility
str_time_now = _deprecated(now, 'mlcrate.time.str_time_now', 'mlcrate.time.now')

def format_duration(seconds, max_fields=3):
    """Formats a number of seconds in a pretty readable format, in terms of seconds, minutes, hours and days.
    Example:
    >>> format_duration(3825.21)
    '1h03m45s'
    >>> format_duration(3825.21, max_fields=2)
    '1h03m'

    Keyword arguments:
    seconds -- A duration to be nicely formatted, in seconds
    max_fields (default: 3) -- The number of units to display (eg. if max_fields is 1 and the time is three days it will only display the days unit)

    Returns: A string representing the duration
    """
    seconds = float(seconds)
    s = int(seconds % 60)
    m = int((seconds / 60) % 60)
    h = int((seconds / 3600) % 24)
    d = int(seconds / 86400)

    fields = []
    for unit, value in zip(['d', 'h', 'm', 's'], [d, h, m, s]):
        if len(fields) > 0: # If it's not the first value, pad with 0s
            fields.append('{}{}'.format(str(value).rjust(2, '0'), unit))
        elif value > 0: # If there are no existing values, we don't add this unit unless it's >0
            fields.append('{}{}'.format(value, unit))

    fields = fields[:max_fields]

    # If the time was less than a second, we just return '<1s' TODO: Maybe return ms instead?
    if len(fields) == 0:
        fields.append('<1s')

    return ''.join(fields)

# Helper Functions
# ---------------------
##avito imp img feats ref...
###    https://www.kaggle.com/sukhyun9673/extracting-image-features-test/log
### https://www.kaggle.com/c/avito-demand-prediction/discussion/59414

def keyp(petID, train = True):
    try:
        if train:
            img = f'../input/petfinder-adoption-prediction/train_images/{petID}-1.jpg'
        else:
            img = f'../input/petfinder-adoption-prediction/test_images/{petID}-1.jpg'
        #only for -1 images
        img = cv2.imread(img, 0)
        fast = cv2.FastFeatureDetector_create()
        kp = fast.detect(img, None)
        kp = len(kp)
        return kp
    except:
        print('exception raised...')
        return 0

class mdict(dict):

    def __setitem__(self, key, value):
        """add the given value to the list of values for this key"""
        self.setdefault(key, []).append(value)

@contextmanager
def faith(title):
    start_time = time.time()
    yield
    print(">> {} - done in {:.0f}s".format(title, time.time() - start_time))

def reduce_mem_usage(df, verbose=True):
    numerics = ['uint8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage(deep=True).sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage(deep=True).sum() / 1024 ** 2
    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem,100 * (start_mem - end_mem) / start_mem))
    return df

def clean_name(x):
    x = str(x)
    no_names = ['None','No Names Yet',"No Name Yet",'Not Yet Named','Noname','Not Named Yet',
    "Nameless",'None Yet','No Named','No Name 2','Not Yet Name',"no_Name_Yet","No Name Yet God Bless",
    "-no Name-","[No Name]","(No Name)","No Names","Not Yet Named"
    ]
    # no_names = ["No Name Yet", "Nameless", "no_Name_Yet", "No Name Yet God Bless", "-no Name-", "[No Name]",
    #             "(No Name)", "No Names", "Not Yet Named"]
    for n in no_names:
        x.replace(n, "No Name")
    return x

def relative_age(cols):
    pet_type = cols[0]
    age = cols[1]
    if pet_type == 1:
        relage = age / 144  # Dog Avergae Life Span - 12 years
    else:
        relage = age / 180  # Cat Average Span - 15 years
    return relage

def VerifibalePhotoAmy(number):
    if number > 1:
        vfp = 1
    else:
        vfp = 0
    return vfp

def seo_value(cols):
    photos = cols[0]
    videos = cols[1]
    seo = .4 * videos + .6 * photos
    return seo

def genuine_name(cols):
    name = cols[0]
    quantity = cols[1]
    try:
        is_gen = int(len(name.split()) == 1)
    except:
        is_gen = np.nan
    if int(quantity) > 1:
        is_gen = 1
    return is_gen

def rankbyG(alldata, group):
    rank_telemetry = pd.DataFrame()
    for unit in (alldata[group].unique()):
        tf = alldata[alldata[group] == unit][['PetID', 'InstaFeature', group]]
        col_name = "Insta" + str(group).title() + "Rank"
        tf[col_name] = tf['InstaFeature'].rank(method='max')
        rank_telemetry = pd.concat([rank_telemetry, tf[['PetID', col_name]]])
        del tf
    alldata = pd.merge(alldata, rank_telemetry, on=['PetID'], how='left')
    return alldata

def get_new_columns(name, aggs):
    return [name + '_' + k + '_' + agg for k in aggs.keys() for agg in aggs[k]]

def agg_features(df, groupby, agg, prefix):
    agg_df = df.groupby(groupby).agg(agg)
    agg_df.columns = get_new_columns(prefix, agg)
    return agg_df

def bounding_features(df, meta_path="../input/petfinder-adoption-prediction/train_metadata/"):
    
    df_id = df['PetID']
    vertex_xs = []
    vertex_ys = []
    bounding_confidences = []
    bounding_importance_fracs = []
    dominant_blues = []
    dominant_greens = []
    dominant_reds = []
    dominant_pixel_fracs = []
    dominant_scores = []
    label_descriptions = []
    label_scores = []
    nf_count = 0
    nl_count = 0
    for pet in (df_id):
        try:
            with open(str(meta_path) + pet + '-1.json', 'r') as f: #adapting to all files didn't help either
                data = json.load(f)
            vertex_x = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['x']
            vertex_xs.append(vertex_x)
            vertex_y = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['y']
            vertex_ys.append(vertex_y)
            bounding_confidence = data['cropHintsAnnotation']['cropHints'][0]['confidence']
            bounding_confidences.append(bounding_confidence)
            bounding_importance_frac = data['cropHintsAnnotation']['cropHints'][0].get('importanceFraction', -1)
            bounding_importance_fracs.append(bounding_importance_frac)
            dominant_blue = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['blue']
            dominant_blues.append(dominant_blue)
            dominant_green = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['green']
            dominant_greens.append(dominant_green)
            dominant_red = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['red']
            dominant_reds.append(dominant_red)
            dominant_pixel_frac = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['pixelFraction']
            dominant_pixel_fracs.append(dominant_pixel_frac)
            dominant_score = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['score']
            dominant_scores.append(dominant_score)
            if data.get('labelAnnotations'):
                label_description = data['labelAnnotations'][0]['description']
                label_descriptions.append(label_description)
                label_score = data['labelAnnotations'][0]['score']
                label_scores.append(label_score)
            else:
                nl_count += 1
                label_descriptions.append('nothing')
                label_scores.append(-1)
        except FileNotFoundError:
            nf_count += 1
            vertex_xs.append(-1)
            vertex_ys.append(-1)
            bounding_confidences.append(-1)
            bounding_importance_fracs.append(-1)
            dominant_blues.append(-1)
            dominant_greens.append(-1)
            dominant_reds.append(-1)
            dominant_pixel_fracs.append(-1)
            dominant_scores.append(-1)
            label_descriptions.append('nothing')
            label_scores.append(-1)
    
    df.loc[:, 'vertex_x'] = vertex_xs
    df.loc[:, 'vertex_y'] = vertex_ys
    df.loc[:, 'bounding_confidence'] = bounding_confidences
    df.loc[:, 'bounding_importance'] = bounding_importance_fracs
    df.loc[:, 'dominant_blue'] = dominant_blues
    df.loc[:, 'dominant_green'] = dominant_greens
    df.loc[:, 'dominant_red'] = dominant_reds
    df.loc[:, 'dominant_pixel_frac'] = dominant_pixel_fracs
    df.loc[:, 'dominant_score'] = dominant_scores
    # df.loc[:, 'label_description'] = label_descriptions
    df.loc[:, 'label_score'] = label_scores
    return df

def bounding_features_all_files(df, meta_path="../input/petfinder-adoption-prediction/train_metadata/*.json"):
    tf = pd.DataFrame()
    pet_ids = []
    vertex_xs = []
    vertex_ys = []
    bounding_confidences = []
    bounding_importance_fracs = []
    dominant_blues = []
    dominant_greens = []
    dominant_reds = []
    dominant_pixel_fracs = []
    dominant_scores = []
    label_descriptions = []
    label_scores = []
    nf_count = 0
    nl_count = 0
    for file in glob.glob(meta_path):
        pet_id = file.split('/')[-1].split('-')[0]
        pet_ids.append(pet_id)
        try:
            with open(str(file), 'r') as f:
                data = json.load(f)
                vertex_x = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['x']
                vertex_xs.append(vertex_x)
                vertex_y = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['y']
                vertex_ys.append(vertex_y)
                bounding_confidence = data['cropHintsAnnotation']['cropHints'][0]['confidence']
                bounding_confidences.append(bounding_confidence)
                bounding_importance_frac = data['cropHintsAnnotation']['cropHints'][0].get('importanceFraction', -1)
                bounding_importance_fracs.append(bounding_importance_frac)
                try:
                    dominant_blue = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['blue']
                    dominant_blues.append(dominant_blue)
                except: dominant_blues.append(-1)
                try:
                    dominant_green = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['green']
                    dominant_greens.append(dominant_green)
                except: dominant_greens.append(-1)
                try:
                    dominant_red = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['red']
                    dominant_reds.append(dominant_red)
                except: dominant_reds.append(-1)
                dominant_pixel_frac = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['pixelFraction']
                dominant_pixel_fracs.append(dominant_pixel_frac)
                dominant_score = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['score']
                dominant_scores.append(dominant_score)
            if data.get('labelAnnotations'):
                label_description = data['labelAnnotations'][0]['description']
                label_descriptions.append(label_description)
                label_score = data['labelAnnotations'][0]['score']
                label_scores.append(label_score)
            else:
                nl_count += 1
                label_descriptions.append('nothing')
                label_scores.append(-1)
        except FileNotFoundError:
            nf_count += 1
            vertex_xs.append(-1)
            vertex_ys.append(-1)
            bounding_confidences.append(-1)
            bounding_importance_fracs.append(-1)
            dominant_blues.append(-1)
            dominant_greens.append(-1)
            dominant_reds.append(-1)
            dominant_pixel_fracs.append(-1)
            dominant_scores.append(-1)
            label_descriptions.append('nothing')
            label_scores.append(-1)

    tf.loc[:,'PetID'] = pet_ids
    tf.loc[:, 'vertex_x_all'] = vertex_xs
    tf.loc[:, 'vertex_y_all'] = vertex_ys
    tf.loc[:, 'bounding_confidence_all'] = bounding_confidences
    tf.loc[:, 'bounding_importance_all'] = bounding_importance_fracs
    tf.loc[:, 'dominant_blue_all'] = dominant_blues
    tf.loc[:, 'dominant_green_all'] = dominant_greens
    tf.loc[:, 'dominant_red_all'] = dominant_reds
    tf.loc[:, 'dominant_pixel_frac_all'] = dominant_pixel_fracs
    tf.loc[:, 'dominant_score_all'] = dominant_scores
    tf.loc[:, 'label_score_all'] = label_scores
    
    return tf

def open_breeds_info_file(filename):
    with open(filename, 'r') as f:
        breedsdata_file = json.load(f)
    return breedsdata_file

def resize_to_square(im, img_size):
    old_size = im.shape[:2]  # old_size is in (height, width) format
    ratio = float(img_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    # new_size should be in (width, height) format
    im = cv2.resize(im, (new_size[1], new_size[0]))
    delta_w = img_size - new_size[1]
    delta_h = img_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return new_im

def load_image(path):
    image = cv2.imread(path).astype(np.float32)
    new_image = resize_to_square(image)
    new_image = preprocess_input(new_image)
    return new_image

def load_image2(path, image_size):
    image = cv2.imread(path).astype(np.float32)
    new_image = resize_to_square(image, image_size)
    new_image = preprocess_input(new_image)
    return new_image

def getSize(filename):
    st = os.stat(filename)
    return st.st_size

def getDimensions(filename):
    img_size = PILImage.open(filename).size
    return img_size

def meta_nlp_feats(df,col):
    
    df[col] = df[col].fillna("None")
    df['length'] = df[col].apply(lambda x : len(x))
    df['capitals'] = df[col].apply(lambda comment: sum(1 for c in comment if c.isupper()))
    df['caps_vs_length'] = df.apply(lambda row: float(row['capitals'])/float(row['length']),axis=1)
    df['num_exclamation_marks'] = df[col].apply(lambda comment: comment.count('!'))
    df['num_question_marks'] = df[col].apply(lambda comment: comment.count('?'))
    df['num_punctuation'] = df[col].apply(lambda comment: sum(comment.count(w) for w in '.,;:'))
    df['num_symbols'] = df[col].apply(lambda comment: sum(comment.count(w) for w in '*&$%'))
    df['num_words'] = df[col].apply(lambda comment: len(comment.split()))
    df['num_unique_words'] = df[col].apply(lambda comment: len(set(w for w in comment.split())))
    df['words_vs_unique'] = df['num_unique_words'] / df['num_words']
    df['num_smilies'] = df[col].apply(lambda comment: sum(comment.count(w) for w in (':-)', ':)', ';-)', ';)')))
    df['num_sad'] = df[col].apply(lambda comment: sum(comment.count(w) for w in (':-<', ':()', ';-()', ';(')))
    
    return df

def text_clean_wrapper(df):
    
    df["Description"] = df["Description"].astype('str').apply(preprocess)
    df['Name'] = df['Name'].astype('str').apply(preprocess)
    # df['sentiment_entities'] = df['sentiment_entities'].astype('str').apply(preprocess)
    # df['metadata_annots_top_desc'] = df['metadata_annots_top_desc'].astype('str').apply(preprocess)
    
    return df
# ============================== PROCESS IN ORDER ===========================

def load_tabular_data():
    
    print('Loading Train Test...')
    train = pd.read_csv('../input/petfinder-adoption-prediction/train/train.csv')
    test = pd.read_csv('../input/petfinder-adoption-prediction/test/test.csv')
    
    label_metadata = {}
    labels_breed = pd.read_csv('../input/petfinder-adoption-prediction/breed_labels.csv')
    labels_color = pd.read_csv('../input/petfinder-adoption-prediction/color_labels.csv')
    labels_state = pd.read_csv('../input/petfinder-adoption-prediction/state_labels.csv')
    
    print('Loaded labels data..')
    ## Using the Kernel:https://www.kaggle.com/bibek777/stacking-kernels
    # state GDP: https://en.wikipedia.org/wiki/List_of_Malaysian_states_by_GDP
    state_gdp = {
        41336: 116.679,
        41325: 40.596,
        41367: 23.02,
        41401: 190.075,
        41415: 5.984,
        41324: 37.274,
        41332: 42.389,
        41335: 52.452,
        41330: 67.629,
        41380: 5.642,
        41327: 81.284,
        41345: 80.167,
        41342: 121.414,
        41326: 280.698,
        41361: 32.270
    }
    # state population: https://en.wikipedia.org/wiki/Malaysia
    state_population = {
        41336: 33.48283,
        41325: 19.47651,
        41367: 15.39601,
        41401: 16.74621,
        41415: 0.86908,
        41324: 8.21110,
        41332: 10.21064,
        41335: 15.00817,
        41330: 23.52743,
        41380: 2.31541,
        41327: 15.61383,
        41345: 32.06742,
        41342: 24.71140,
        41326: 54.62141,
        41361: 10.35977
    }
    state_area ={
        41336:19102,
        41325:9500,
        41367:15099,
        41401:243,
        41415:91,
        41324:1664,
        41332:6686,
        41335:36137,
        41330:21035,
        41380:821,
        41327:1048,
        41345:73631,
        41342:124450,
        41326:8104,
        41361:13035
    }
    
    # https://www.dosm.gov.my/
    # Unemployment Rate in 2017
    state_unemployment ={
        41336 : 3.6,
        41325 :2.9,
        41367: 3.8,
        41324: 0.9,
        41332 : 2.7,
        41335: 2.6,
        41330: 3.4,
        41380: 2.9,
        41327: 2.1,
        41345 : 5.4,
        41342 : 3.3,
        41326: 3.2,
        41361: 4.2,
        41415: 7.8,
        41401: 3.3
    }
    # https://www.dosm.gov.my/
    # per 1000 population in 2016
    state_birth_rate = {
        41336:16.3,
        41325:17.0,
        41367:21.4,
        41401:14.4,
        41415:18.1,
        41324:16.0,
        41332:16.4,
        41335:17.0,
        41330:14.4,
        41380:17.5,
        41327:12.7,
        41345:13.7,
        41342:13.9,
        41326:16.6,
        41361:23.3,     
    }
    
    train["state_gdp"] = train.State.map(state_gdp)
    train["state_population"] = train.State.map(state_population)
    train["state_area"] = train.State.map(state_area)
    train['state_unemployment']=train.State.map(state_unemployment)
    train['state_birth_rate']=train.State.map(state_birth_rate)
    
    test["state_gdp"] =test.State.map(state_gdp)
    test["state_population"] = test.State.map(state_population)
    test["state_area"] = test.State.map(state_area)
    test['state_unemployment']=test.State.map(state_unemployment)
    test['state_birth_rate']=test.State.map(state_birth_rate)
    
    print('Running Text Cleaning')
    
    train = df_parallelize_run(train, text_clean_wrapper)
    test  = df_parallelize_run(test, text_clean_wrapper)
    
    print('Calculating keyp...')
    
    dfs_train = Parallel(n_jobs=4, verbose=1)(delayed(keyp)(i, train = True) for i in train['PetID'].values.tolist())
    dfs_test  = Parallel(n_jobs=4, verbose=1)(delayed(keyp)(i, train = False) for i in test['PetID'].values.tolist())
    
    train['keyp_val'] = dfs_train
    test['keyp_val'] = dfs_test
    
    del dfs_train, dfs_test
    gc.collect()
    # train.drop('Breed', axis=1, inplace=True)
    return train, test, labels_state, labels_breed, labels_color

def load_image_data():
    
    train_image_files = glob.glob('../input/petfinder-adoption-prediction/train_images/*.jpg')
    test_image_files = glob.glob('../input/petfinder-adoption-prediction/test_images/*.jpg')

    train_df_imgs = pd.DataFrame(train_image_files)
    train_df_imgs.columns = ['image_filename']
    train_imgs_pets = train_df_imgs['image_filename'].apply(lambda x: x.split('/')[-1].split('-')[0])
    train_df_imgs = train_df_imgs.assign(PetID=train_imgs_pets)

    test_df_imgs = pd.DataFrame(test_image_files)
    test_df_imgs.columns = ['image_filename']
    test_imgs_pets = test_df_imgs['image_filename'].apply(lambda x: x.split('/')[-1].split('-')[0])
    test_df_imgs = test_df_imgs.assign(PetID=test_imgs_pets)

    return train_df_imgs, test_df_imgs

def load_metadata():
    
    train_metadata_files = glob.glob('../input/petfinder-adoption-prediction/train_metadata/*.json')
    test_metadata_files = glob.glob('../input/petfinder-adoption-prediction/test_metadata/*.json')

    train_df_metadata = pd.DataFrame(train_metadata_files)
    train_df_metadata.columns = ['metadata_filename']
    train_metadata_pets = train_df_metadata['metadata_filename'].apply(lambda x: x.split('/')[-1].split('-')[0])
    train_df_metadata = train_df_metadata.assign(PetID=train_metadata_pets)

    test_df_metadata = pd.DataFrame(test_metadata_files)
    test_df_metadata.columns = ['metadata_filename']
    test_metadata_pets = test_df_metadata['metadata_filename'].apply(lambda x: x.split('/')[-1].split('-')[0])
    test_df_metadata = test_df_metadata.assign(PetID=test_metadata_pets)

    return train_df_metadata, test_df_metadata

def load_sentiment_data():
    train_sentiment_files = glob.glob('../input/petfinder-adoption-prediction/train_sentiment/*.json')
    test_sentiment_files = glob.glob('../input/petfinder-adoption-prediction/test_sentiment/*.json')

    train_df_sentiment = pd.DataFrame(train_sentiment_files)
    train_df_sentiment.columns = ['sentiment_filename']
    train_sentiment_pets = train_df_sentiment['sentiment_filename'].apply(lambda x: x.split('/')[-1].split('.')[0])
    train_df_sentiment = train_df_sentiment.assign(PetID=train_sentiment_pets)

    test_df_sentiment = pd.DataFrame(test_sentiment_files)
    test_df_sentiment.columns = ['sentiment_filename']
    test_sentiment_pets = test_df_sentiment['sentiment_filename'].apply(lambda x: x.split('/')[-1].split('.')[0])
    test_df_sentiment = test_df_sentiment.assign(PetID=test_sentiment_pets)

    return train_df_sentiment, test_df_sentiment

def build_model_img(shape=(256, 256, 3), weights_path="../input/densenet-keras/DenseNet-BC-121-32-no-top.h5", resnet= False):
    
    from keras.models import Model
    from keras.preprocessing.text import Tokenizer
    from keras.models import Model
    from keras.layers import GlobalAveragePooling2D, Input, Lambda, AveragePooling1D, CuDNNLSTM, CuDNNGRU
    from keras.preprocessing.sequence import pad_sequences
    from keras.layers import Dense, Input, CuDNNLSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D
    from keras.layers import Bidirectional, GlobalMaxPool1D, GlobalMaxPooling2D, GlobalAveragePooling1D, GlobalAvgPool2D, GlobalMaxPool2D
    from keras.layers import Input, Embedding, Dense, Conv2D, MaxPool2D, concatenate
    from keras.layers import Reshape, Flatten, Concatenate, Dropout, SpatialDropout1D, SpatialDropout2D
    from keras.applications.densenet import preprocess_input, DenseNet121
    from keras.optimizers import Adam
    from keras.models import Model
    from keras import backend as K
    from keras.engine.topology import Layer
    from keras import initializers, regularizers, constraints, optimizers, layers
    if resnet:
        print('Using Resnet50')
        inp = Input(shape)
        backbone = ResNet50(include_top=False,
                     weights= '../input/keras-pretrained-models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                     input_shape= shape,
                     pooling='avg')
        x = Lambda(lambda x: K.expand_dims(x, axis=-1))(x)
        x = AveragePooling1D(4)(x)
        out = Lambda(lambda x: x[:, :, 0])(x)
        model = Model(inp, out)
    else:
        print('Using DenseNet')
        inp = Input(shape)
        backbone = DenseNet121(input_tensor=inp,
                              weights=weights_path,
                              include_top=False)
        x = backbone.output
        x = GlobalAveragePooling2D()(x)
        x = Lambda(lambda x: K.expand_dims(x, axis=-1))(x)
        x = AveragePooling1D(4)(x)
        out = Lambda(lambda x: x[:, :, 0])(x)
        model = Model(inp, out)
        
    return model

def train_model(model, train, test, nn_params={"batch_size": 128, "img_size": 256}):
    
    batch_size = nn_params['batch_size']
    img_size = nn_params['img_size']
    pet_ids = train['PetID'].values
    train_df_ids = train[['PetID']]

    # Train images
    features = {}
    train_image = glob.glob("../input/petfinder-adoption-prediction/train_images/*.jpg")
    n_batches = len(train_image) // batch_size + (len(train_image) % batch_size != 0)
    for b in (range(n_batches)):
        start = b * batch_size
        end = (b + 1) * batch_size
        batch_pets = train_image[start:end]
        batch_images = np.zeros((len(batch_pets), img_size, img_size, 3))
        for i, pet_id in enumerate(batch_pets):
            try:
                batch_images[i] = load_image(pet_id)
            except:
                pass
        batch_preds = model.predict(batch_images)
        for i, pet_id in enumerate(batch_pets):
            features[pet_id] = batch_preds[i]

    train_feats = pd.DataFrame.from_dict(features, orient='index')
    train_feats.columns = ['pic_' + str(i) for i in range(train_feats.shape[1])]

    train_feats = train_feats.reset_index()
    train_feats['PetID'] = train_feats['index'].apply(lambda x: x.split("/")[-1].split("-")[0])
    train_feats = train_feats.drop("index", axis=1)
    train_feats = train_feats.groupby('PetID').agg("mean")
    train_feats = train_feats.reset_index()

    # Test images
    features = {}

    test_image = glob.glob("../input/petfinder-adoption-prediction/test_images/*.jpg")
    n_batches = len(test_image) // batch_size + (len(test_image) % batch_size != 0)
    for b in (range(n_batches)):
        start = b * batch_size
        end = (b + 1) * batch_size
        batch_pets = test_image[start:end]
        batch_images = np.zeros((len(batch_pets), img_size, img_size, 3))
        for i, pet_id in enumerate(batch_pets):
            try:
                batch_images[i] = load_image(pet_id)
            except:
                pass
        batch_preds = model.predict(batch_images)
        for i, pet_id in enumerate(batch_pets):
            features[pet_id] = batch_preds[i]

    test_feats = pd.DataFrame.from_dict(features, orient='index')
    test_feats.columns = ['pic_' + str(i) for i in range(test_feats.shape[1])]

    test_feats = test_feats.reset_index()
    test_feats['PetID'] = test_feats['index'].apply(lambda x: x.split("/")[-1].split("-")[0])
    test_feats = test_feats.drop("index", axis=1)
    test_feats = test_feats.groupby('PetID').agg("mean")
    test_feats = test_feats.reset_index()
    pretrained_feats = pd.concat([train_feats, test_feats], axis=0)

    return pretrained_feats

def image_feature(model, train, test, nn_params={"batch_size": 128, "img_size": 256}):
    
    if not preload:
        batch_size = nn_params['batch_size']
        img_size = nn_params['img_size']
        train_df_ids = train[['PetID']]

        # Train images
        features = {}
        train_image = glob.glob("../input/petfinder-adoption-prediction/train_images/*.jpg")
        n_batches = len(train_image) // batch_size + (len(train_image) % batch_size != 0)
        for b in range(n_batches):
            start = b * batch_size
            end = (b + 1) * batch_size
            batch_pets = train_image[start:end]
            batch_images = np.zeros((len(batch_pets), img_size, img_size, 3))
            for i, pet_id in enumerate(batch_pets):
                try:
                    batch_images[i] = load_image2(pet_id, img_size)
                except:
                    print(pet_id)
                    pass
            batch_preds = model.predict(batch_images)
            for i, pet_id in enumerate(batch_pets):
                features[pet_id] = batch_preds[i]

        train_feats = pd.DataFrame.from_dict(features, orient='index')
        train_feats.columns = ['pic_' + str(i) for i in range(train_feats.shape[1])]

        train_feats = train_feats.reset_index()
        train_feats['PetID'] = train_feats['index'].apply(lambda x: x.split("/")[-1].split("-")[0])
        train_feats = train_feats.drop("index", axis=1)
        train_feats = train_feats.groupby('PetID').agg("mean")
        train_feats = train_feats.reset_index()

        # Test images
        features = {}

        test_image = glob.glob("../input/petfinder-adoption-prediction/test_images/*.jpg")
        n_batches = len(test_image) // batch_size + (len(test_image) % batch_size != 0)
        for b in (range(n_batches)):
            start = b * batch_size
            end = (b + 1) * batch_size
            batch_pets = test_image[start:end]
            batch_images = np.zeros((len(batch_pets), img_size, img_size, 3))
            for i, pet_id in enumerate(batch_pets):
                try:
                    batch_images[i] = load_image2(pet_id, img_size)
                except:
                    print(pet_id)
                    pass
            batch_preds = model.predict(batch_images)
            for i, pet_id in enumerate(batch_pets):
                features[pet_id] = batch_preds[i]

        test_feats = pd.DataFrame.from_dict(features, orient='index')
        test_feats.columns = ['pic_' + str(i) for i in range(test_feats.shape[1])]

        test_feats = test_feats.reset_index()
        test_feats['PetID'] = test_feats['index'].apply(lambda x: x.split("/")[-1].split("-")[0])
        test_feats = test_feats.drop("index", axis=1)
        test_feats = test_feats.groupby('PetID').agg("mean")
        test_feats = test_feats.reset_index()
        pretrained_feats = pd.concat([train_feats, test_feats], axis=0)
    else:
        train_feats = pd.read_csv("./processed_data/train_img.csv")
        test_feats = pd.read_csv("./processed_data/test_img.csv")
        pretrained_feats = pd.concat([train_feats, test_feats], axis=0)

    return pretrained_feats

def add_sentimental_analysis(df, path_name= '../input/petfinder-adoption-prediction/train_sentiment/*.json', is_train = True):
    
    if is_train:
        flag_id = 'train'
    else:
        flag_id = 'test'
    
    sentimental_analysis = glob.glob(path_name)
    score=[]
    magnitude=[]
    petid=[]
    for filename in sentimental_analysis:
        with open(filename, 'r') as f:
            sentiment_file = json.load(f)
        
        file_sentiment = sentiment_file['documentSentiment']
        file_score =  np.asarray(sentiment_file['documentSentiment']['score'])
        file_magnitude = np.asarray(sentiment_file['documentSentiment']['magnitude'])
        score.append(file_score)
        magnitude.append(file_magnitude)
        petid.append(filename.replace('.json','').replace(f'../input/petfinder-adoption-prediction/{flag_id}_sentiment/', ''))
    #create a df    
    sentimental_analysis = pd.concat([pd.DataFrame(petid, columns =['PetID']) ,pd.DataFrame(score, columns =['sentiment_document_score']),
    pd.DataFrame(magnitude, columns =['sentiment_document_magnitude'])],axis =1)
    #merge
    df = pd.merge(df, sentimental_analysis, how='left', on='PetID')
    
    return df

def add_image_fe(df, path_name = '../input/petfinder-adoption-prediction/train_images/*.jpg', is_train=True):
    
    print('Calculating Image Quality Random Feats.....')
    
    if is_train:
        flag_id = 'train'
    else:
        flag_id = 'test'
    
    image_quality = glob.glob(path_name)
    
    blur=[]
    image_pixel=[]
    imageid =[]
    
    for filename in image_quality:
        #Blur 
        image = cv2.imread(filename)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        result = cv2.Laplacian(gray, cv2.CV_64F).var()
        # Pixels
        with PILImage.open(filename) as pixel:
            width, height = pixel.size
        pixel = width*height
        image_pixel.append(pixel)
        #blur for each image
        blur.append(result)
        #image id
        imageid.append(filename.replace('.jpg','').replace(f'../input/petfinder-adoption-prediction/{flag_id}_images/', ''))
    
    # Join Pixel, Blur and Image ID #currently blur has been removed for speed concerns..
    image_quality = pd.concat([pd.DataFrame(imageid, columns =['ImageId']),pd.DataFrame(image_pixel,columns=['pixel']),pd.DataFrame(blur,columns=['blur'])],axis =1)
    # create the PetId variable
    image_quality['PetID'] = image_quality['ImageId'].str.split('-').str[0]
    #Mean of the Mean
    image_quality['pixel_mean'] = image_quality.groupby(['PetID'])['pixel'].transform('mean')
    image_quality['blur_mean'] = image_quality.groupby(['PetID'])['blur'].transform('mean') 
    
    image_quality['pixel_min'] = image_quality.groupby(['PetID'])['pixel'].transform('min') 
    # image_quality['blur_min'] = image_quality.groupby(['PetID'])['blur'].transform('min')
    
    image_quality['pixel_max'] = image_quality.groupby(['PetID'])['pixel'].transform('max') 
    # image_quality['blur_max'] = image_quality.groupby(['PetID'])['blur'].transform('max')
    
    image_quality = image_quality.drop(['pixel','ImageId'], 1)
    image_quality = image_quality.drop_duplicates('PetID')
    
    df = pd.merge(df, image_quality,  how='left', left_on=['PetID'], right_on = ['PetID'])
    del image_quality
    gc.collect()
    print('Done Calculating Image Quality....')
    
    return df

def basic_features(train, test):
    
    try:
        print('fast.ai version:{}'.format(fastai.__version__))
        print('Working On CuteNess Be Patient.... haha..')
        #save all paths -1
        pd.DataFrame(glob.glob('../input/petfinder-adoption-prediction/train_images/*-1.jpg')).to_csv('tr_img.csv', index=None)
        pd.DataFrame(glob.glob('../input/petfinder-adoption-prediction/test_images/*.jpg')).to_csv('test_img.csv', index=None) #dont remove the *
        #fastai learner
        learn = load_learner('.', '../input/cuteness-fastai/trained_model_cuteness.pkl',test=ImageList.from_csv('.', 'tr_img.csv'))
        learn.data.batch_size = 18 # DON'T INCREASE THIS REDUCE ALWAYS WONT HURT
        preds_tr,_ = learn.get_preds(ds_type=DatasetType.Test) #takes 2-3 mins on -1, overall 8 mins for all -id files if want but need to agg then
        tr_img = pd.read_csv('tr_img.csv').rename(columns={'0':'PetID'})
        tr_img['PetID'] = tr_img['PetID'].apply(lambda x: x.split('/')[-1].split('-')[0])
        tr_img['preds'] = preds_tr[:,1].numpy()
        tr_img['preds'] = tr_img['preds'].astype(float)
        # tr_img = tr_img.groupby('PetID', sort=False).agg('mean').reset_index()
        train = pd.merge(train, tr_img, how='left', on=['PetID'])
        del learn, preds_tr, tr_img
        gc.collect()
        
        print('Test Cuteness')
        learn = load_learner('.', '../input/cuteness-fastai/trained_model_cuteness.pkl',test=ImageList.from_csv('.', 'test_img.csv'))
        learn.data.batch_size = 18 # DON'T INCREASE THIS REDUCE ALWAYS WONT HURT
        preds_tr,_ = learn.get_preds(ds_type=DatasetType.Test) #this is fast
        test_img = pd.read_csv('test_img.csv').rename(columns={'0':'PetID'})
        test_img['PetID'] = test_img['PetID'].apply(lambda x: x.split('/')[-1].split('-')[0])
        test_img['preds'] = preds_tr[:,1].numpy()
        test_img['preds'] = test_img['preds'].astype(float)
        test_img = test_img.groupby('PetID', sort=False).agg('mean').reset_index()
        test  = pd.merge(test, test_img, how='left', on=['PetID'])
        
        del learn, preds_tr, test_img
        gc.collect()
        
    except:
        print('couldnt add cuteness feats....')
        
    alldata = pd.concat([train, test], sort=False)
    print(train.shape, test.shape, alldata.shape)
    ################################ NEW STUFF TESTING
    Breed1_count = alldata.groupby('Breed1').size().to_frame('Breed1_count').reset_index()
    alldata = alldata.merge(Breed1_count, how='left', on='Breed1')
    
    a = alldata['Breed1'].value_counts().sort_values(ascending = False).cumsum()/len(alldata)
    rare1_index = a[a > 0.85].index.tolist()
    alldata['IsRare1'] = alldata['Breed1'].isin(rare1_index).apply(lambda x:1 if x == True else 0)
    
    rare2_index = a[a > 0.72].index.tolist()
    alldata['IsRare2'] = alldata['Breed1'].isin(rare2_index).apply(lambda x:1 if x == True else 0)
    alldata['Is_COMMON'] = alldata['Breed1'].apply(lambda x:1 if (x == 265 or x == 307 or x == 266) else 0)
    
    alldata['cat_less_than_5'] = 0
    alldata.loc[alldata[(alldata['Type'] == 2) & (alldata['Age'] <= 5)].index, 'cat_less_than_5'] = 1
    
    alldata['dog_less_than_5'] = 0
    alldata.loc[alldata[(alldata['Type'] == 1) & (alldata['Age'] <= 5)].index, 'dog_less_than_5'] = 1
    
    alldata['cat_less_than_12_white'] = 0
    alldata.loc[alldata[(alldata['Type'] == 2) & (alldata['Age'] <= 12) & (alldata['Color1'] == 7)].index, 'cat_less_than_12_white'] = 1
    
    alldata['dog_less_than_12_black'] = 0
    alldata.loc[alldata[(alldata['Type'] == 1) & (alldata['Age'] <= 12) & (alldata['Color1'] == 1)].index, 'dog_less_than_12_black'] = 1
    
    ####### dog grouping classes
    d = {}
    for idx,row in enumerate(labels_breed[['BreedID', 'BreedName']].values):
        d[row[1]] = int(row[0])
    z = []
    short = ['Australian Shepherd','Boston Terrier','Brittany Spaniel','English Bulldog','French Bulldog','Jack Russell Terrier','Karelian Bear Dog','McNab','Munsterlander','Old English Sheepdog','Polish Lowland Sheepdog','Rat Terrier','Rottweiler','Schipperke','Swedish Vallhund','Bobtail','Cymric','Japanese Bobtail','Manx','Pixie-Bob','Siamese']
    for i in d.keys():
        if i in short:
            z.append(d[i])
    del short

    alldata['bob_tails'] = 0
    alldata.loc[alldata['Breed1'].isin(z)==True, 'bob_tails'] = 1

    alldata['bob_tails_health_1'] = 0
    alldata.loc[alldata[(alldata['bob_tails'] == 1) & (alldata['Health'] == 1)].index, 'bob_tails_health_1'] = 1
    alldata['bob_tails_health_2'] = 0
    alldata.loc[alldata[(alldata['bob_tails'] == 1) & (alldata['Health'] == 2)].index, 'bob_tails_health_2'] = 1
    alldata['bob_tails_health_3'] = 0
    alldata.loc[alldata[(alldata['bob_tails'] == 1) & (alldata['Health'] == 3)].index, 'bob_tails_health_3'] = 1
    
    del z
    
    breeds = ['Welsh Corgi','Kai Dog','Lowchen','English Pointer','Setter','White German Shepherd','German Spitz','Saint Bernard','Jack Russell Terrier (Parson Russell Terrier)','Spitz','Mixed Breed','Terrier','Spaniel','Dutch Shepherd','Foxhound','Lancashire Heeler','Cattle Dog','Scottish Terrier Scottie','Fox Terrier','Flat-coated Retriever','Wirehaired Terrier','Shetland Sheepdog Sheltie','White German Shepherd,''German Spitz','Sheep Dog','Hound','Yorkshire Terrier Yorkie','Coonhound','Mountain Dog','Munsterlander','Wheaten Terrier','West Highland White Terrier Westie','Yellow Labrador Retriever','Chocolate Labrador Retriever','Belgian Shepherd Dog Sheepdog','Corgi','Shepherd','Jack Russell Terrier','Belgian Shepherd Laekenois','Belgian Shepherd Malinois','Shar Pei','Black Labrador Retriever','Poodle','German Shepherd Dog','Schnauzer','Husky','Pit Bull Terrier','Jack Russell Terrier','Cattle Dog','Blue Lacy','Queensland Heeler','Rhod Ridgeback','Retriever','Chinese Sharpei','Black Mouth Cur','Catahoula','Staffordshire','Affenpinscher','Afghan Hound','Airedale Terrier','Akita','Australian Kelpie','Alaskan Malamute','English Bulldog','American Bulldog','American English Coonhound','American Eskimo Dog (Miniature)','American Eskimo Dog (Standard)','American Eskimo Dog (Toy)','American Foxhound','American Hairless Terrier','American Staffordshire Terrier','American Water Spaniel','Anatolian Shepherd Dog','Australian Cattle Dog','Australian Shepherd','Australian Terrier','Basenji','Basset Hound','Beagle','Bearded Collie','Beauceron','Bedlington Terrier','Belgian Malinois','Belgian Sheepdog','Belgian Tervuren','Bergamasco','Berger Picard','Bernese Mountain Dog','Bichon Fris_','Black and Tan Coonhound','Black Russian Terrier','Bloodhound','Bluetick Coonhound','Boerboel','Border Collie','Border Terrier','Borzoi','Boston Terrier','Bouvier des Flandres','Boxer','Boykin Spaniel','Briard','Brittany','Brussels Griffon','Bull Terrier','Bull Terrier (Miniature)','Bulldog','Bullmastiff','Cairn Terrier','Canaan Dog','Cane Corso','Cardigan Welsh Corgi','Cavalier King Charles Spaniel','Cesky Terrier','Chesapeake Bay Retriever','Chihuahua','Chinese Crested Dog','Chinese Shar Pei','Chinook','Chow Chow',"Cirneco dell'Etna",'Clumber Spaniel','Cocker Spaniel','Collie','Coton de Tulear','Curly-Coated Retriever','Dachshund','Dalmatian','Dandie Dinmont Terrier','Doberman Pinsch','Doberman Pinscher','Dogue De Bordeaux','English Cocker Spaniel','English Foxhound','English Setter','English Springer Spaniel','English Toy Spaniel','Entlebucher Mountain Dog','Field Spaniel','Finnish Lapphund','Finnish Spitz','Flat-Coated Retriever','French Bulldog','German Pinscher','German Shepherd','German Shorthaired Pointer','German Wirehaired Pointer','Giant Schnauzer','Glen of Imaal Terrier','Golden Retriever','Gordon Setter','Great Dane','Great Pyrenees','Greater Swiss Mountain Dog','Greyhound','Harrier','Havanese','Ibizan Hound','Icelandic Sheepdog','Irish Red and White Setter','Irish Setter','Irish Terrier','Irish Water Spaniel','Irish Wolfhound','Italian Greyhound','Japanese Chin','Keeshond','Kerry Blue Terrier','Komondor','Kuvasz','Labrador Retriever','Lagotto Romagnolo','Lakeland Terrier','Leonberger','Lhasa Apso','L_wchen','Maltese','Manchester Terrier','Mastiff','Miniature American Shepherd','Miniature Bull Terrier','Miniature Pinscher','Miniature Schnauzer','Neapolitan Mastiff','Newfoundland','Norfolk Terrier','Norwegian Buhund','Norwegian Elkhound','Norwegian Lundehund','Norwich Terrier','Nova Scotia Duck Tolling Retriever','Old English Sheepdog','Otterhound','Papillon','Parson Russell Terrier','Pekingese','Pembroke Welsh Corgi','Petit Basset Griffon Vend_en','Pharaoh Hound','Plott','Pointer','Polish Lowland Sheepdog','Pomeranian','Standard Poodle','Miniature Poodle','Toy Poodle','Portuguese Podengo Pequeno','Portuguese Water Dog','Pug','Puli','Pyrenean Shepherd','Rat Terrier','Redbone Coonhound','Rhodesian Ridgeback','Rottweiler','Russell Terrier','St. Bernard','Saluki','Samoyed','Schipperke','Scottish Deerhound','Scottish Terrier','Sealyham Terrier','Shetland Sheepdog','Shiba Inu','Shih Tzu','Siberian Husky','Silky Terrier','Skye Terrier','Sloughi','Smooth Fox Terrier','Soft-Coated Wheaten Terrier','Spanish Water Dog','Spinone Italiano','Staffordshire Bull Terrier','Standard Schnauzer','Sussex Spaniel','Swedish Vallhund','Tibetan Mastiff','Tibetan Spaniel','Tibetan Terrier','Toy Fox Terrier','Treeing Walker Coonhound','Vizsla','Weimaraner','Welsh Springer Spaniel','Welsh Terrier','West Highland White Terrier','Whippet','Wire Fox Terrier','Wirehaired Pointing Griffon','Wirehaired Vizsla','Xoloitzcuintli','Yorkshire Terrier']
    groups = ['Herding','Working','Non-Sporting','Sporting','Sporting','Herding','Non-Sporting','Working','Terrier','Non-Sporting','Miscellaneous','Terrier','Sporting','Herding','Hound','Miscellaneous','Herding','Terrier','Terrier','Working','Terrier','Herding','Herding','Non-Sporting','Herding','Hound','Terrier','Hound','Working','Hunting','Terrier','Terrier','Working','Working','Terrier','Herding','Herding','Herding','Terrier','Herding','Herding','Non-Sporting','Working','Non-Sporting','Herding','Working','Working','Sporting','Terrier','Herding','Herding','Herding','Hound','Sporting','Non-Sporting','Herding','Herding','Terrier','Toy','Hound','Terrier','Working','Working','Working','Non-Sporting','Non-Sporting','Hound','Non-Sporting','Non-Sporting','Toy','Hound','Terrier','Terrier','Sporting','Working','Herding','Herding','Terrier','Hound','Hound','Hound','Herding','Herding','Terrier','Herding','Herding','Herding','Herding','Herding','Working','Non-Sporting','Hound','Working','Hound','Hound','Working','Herding','Terrier','Hound','Non-Sporting','Herding','Working','Sporting','Herding','Sporting','Toy','Terrier','Terrier','Non-Sporting','Working','Terrier','Working','Working','Herding','Toy','Terrier','Sporting','Toy','Toy','Non-Sporting','Working','Non-Sporting','Hound','Sporting','Sporting','Herding','Non-Sporting','Sporting','Hound','Non-Sporting','Terrier','Working','Working','Working','Sporting','Hound','Sporting','Sporting','Toy','Herding','Sporting','Herding','Non-Sporting','Sporting','Non-Sporting','Working','Herding','Sporting','Sporting','Working','Terrier','Sporting','Sporting','Working','Working','Working','Hound','Hound','Toy','Hound','Herding','Sporting','Sporting','Terrier','Sporting','Hound','Toy','Toy','Non-Sporting','Terrier','Working','Working','Sporting','Sporting','Terrier','Working','Non-Sporting','Non-Sporting','Toy','Terrier','Working','Herding','Terrier','Toy','Terrier','Working','Working','Terrier','Herding','Hound','Non-Sporting','Terrier','Sporting','Herding','Hound','Toy','Terrier','Toy','Herding','Hound','Hound','Hound','Sporting','Herding','Toy','Non-Sporting','Non-Sporting','Toy','Hound','Working','Toy','Herding','Herding','Terrier','Hound','Hound','Working','Terrier','Working','Hound','Working','Non-Sporting','Hound','Terrier','Terrier','Herding','Non-Sporting','Toy','Working','Toy','Terrier','Hound','Terrier','Terrier','Herding','Sporting','Terrier','Working','Sporting','Herding','Working','Non-Sporting','Non-Sporting','Toy','Hound','Sporting','Sporting','Sporting','Terrier','Terrier','Hound','Terrier','Sporting','Sporting','Non-Sporting','Toy']
    
    grp_map = {}
    breed_grouping = {}
    
    for i,j in (zip(breeds, groups)):
        breed_grouping[i] = j
    feature_values_dog = alldata.loc[alldata['Type'] == 1,'Breed1']
    final_map = {}
    
    for i in feature_values_dog.unique():
        if i!=0:
            try:
                final_map[i] = breed_grouping[dict((v, k) for k, v in d.items())[i]]
            except:
                print('Can\'t find this in breed_mapping', dict((v, k) for k, v in d.items())[i])
                
    alldata['group_pets'] = alldata['Breed1'].map(final_map).fillna('cat_class').astype('category')
    del rare1_index, rare2_index, a, Breed1_count, final_map, feature_values_dog, breed_grouping, grp_map, breeds, groups, d
    gc.collect()
    #########################################################################################################
    #### old things intact
    
    # Breed create columns
    alldata['weeks'] = alldata['Age']*31//7
    alldata['shorthair_hairless_domestic_hair'] = 0
    alldata.loc[alldata['Breed1'].isin([9 ,104 ,106 ,236 ,237 ,238 ,243 ,244 ,251 ,255 ,264 ,265 ,266 ,268 ,282 ,283 ,298]) == True, 'shorthair_hairless_domestic_hair'] = 1
    
    temp = alldata[['RescuerID', 'State', 'Type', 'Age']].groupby(['RescuerID', 'State', 'Type'], sort=False)['Age'].agg(['count']).reset_index().sort_values(by=['RescuerID', 'State'])['RescuerID'].value_counts().reset_index()
    temp = temp[temp['RescuerID'] == 2]['index'].values.tolist()
    
    alldata['#Feature_resucers_saved_both_c_d'] = 0
    alldata.loc[alldata['State'].isin(temp) == True, '#Feature_resucers_saved_both_c_d'] = 1
    del temp
    gc.collect()
    
    alldata['#Feature_avg_age_color1_fee'] = alldata[['Age', 'Color1', 'Fee']].groupby(['Age', 'Color1'])['Fee'].transform('mean')
    alldata['#Feature_avg_age_color2_fee'] = alldata[['Age', 'Color2', 'Fee']].groupby(['Age', 'Color2'])['Fee'].transform('mean')
    
    alldata['#Feature_avg_age_breed1_fee'] = alldata[['Age', 'Breed1', 'Fee']].groupby(['Age', 'Breed1'])['Fee'].transform('mean')
    alldata['#Feature_avg_age_breed2_fee'] = alldata[['Age', 'Breed2', 'Fee']].groupby(['Age', 'Breed2'])['Fee'].transform('mean')
    alldata['#Feature_age_breed1_maturity_sz'] = alldata[[ 'Age', 'Breed1', 'MaturitySize']].groupby([ 'Age', 'Breed1'])['MaturitySize'].transform('count') / alldata.shape[0]
    alldata['#Feature_age_breed2_maturity_sz'] = alldata[[ 'Age', 'Breed2', 'MaturitySize']].groupby([ 'Age', 'Breed2'])['MaturitySize'].transform('count') / alldata.shape[0]
    
    alldata['#Feature_age_breed1_fur'] = alldata[[ 'Age', 'Breed1', 'FurLength']].groupby([ 'Age', 'Breed1'])['FurLength'].transform('count') / alldata.shape[0]
    alldata['#Feature_age_breed2_fur'] = alldata[[ 'Age', 'Breed2', 'FurLength']].groupby([ 'Age', 'Breed2'])['FurLength'].transform('count') / alldata.shape[0]
    alldata['#Feature_age_breed1_fee'] = alldata[[ 'Age', 'Breed1', 'Fee']].groupby([ 'Age', 'Breed1'])['Fee'].transform('count') / alldata.shape[0]
    alldata['#Feature_age_breed2_fee'] = alldata[[ 'Age', 'Breed2', 'Fee']].groupby([ 'Age', 'Breed2'])['Fee'].transform('count') / alldata.shape[0]
    
    alldata['#Feature_state_breed1_age_freq']     = alldata[[ 'State', 'Breed1', 'Age']].groupby([ 'State', 'Breed1'])['Age'].transform('mean')
    alldata['#Feature_state_breed1_age_fee_freq'] = alldata[[ 'State', 'Breed1', 'Age', 'Fee']].groupby([ 'State', 'Breed1', 'Age'])['Fee'].transform('mean')
    alldata['#Feature_state_breed2_age_freq']     = alldata[[ 'State', 'Breed2', 'Age']].groupby([ 'State', 'Breed2'])['Age'].transform('mean')
    alldata['#Feature_state_breed2_age_fee_freq'] = alldata[[ 'State', 'Breed2', 'Age', 'Fee']].groupby([ 'State', 'Breed2', 'Age'])['Fee'].transform('mean')
    
    alldata['#Feature_avg_type_age_breed1_fee'] = alldata[['Type','Age', 'Breed1', 'Fee']].groupby(['Type','Age', 'Breed1'])['Fee'].transform('mean')
    alldata['#Feature_avg_type_age_breed2_fee'] = alldata[['Type','Age', 'Breed2', 'Fee']].groupby(['Type','Age', 'Breed2'])['Fee'].transform('mean')
    alldata['#Feature_age_type_breed1_maturity_sz'] = alldata[['Type', 'Age', 'Breed1', 'MaturitySize']].groupby(['Type', 'Age', 'Breed1'])['MaturitySize'].transform('count') / alldata.shape[0]
    alldata['#Feature_age_type_breed2_maturity_sz'] = alldata[['Type', 'Age', 'Breed2', 'MaturitySize']].groupby(['Type', 'Age', 'Breed2'])['MaturitySize'].transform('count') / alldata.shape[0]
    
    alldata['#Feature_age_type_breed1_fur'] = alldata[['Type', 'Age', 'Breed1', 'FurLength']].groupby(['Type', 'Age', 'Breed1'])['FurLength'].transform('count') / alldata.shape[0]
    alldata['#Feature_age_type_breed2_fur'] = alldata[['Type', 'Age', 'Breed2', 'FurLength']].groupby(['Type', 'Age', 'Breed2'])['FurLength'].transform('count') / alldata.shape[0]
    alldata['#Feature_age_type_breed1_fee'] = alldata[['Type', 'Age', 'Breed1', 'Fee']].groupby(['Type', 'Age', 'Breed1'])['Fee'].transform('count') / alldata.shape[0]
    alldata['#Feature_age_type_breed2_fee'] = alldata[['Type', 'Age', 'Breed2', 'Fee']].groupby(['Type', 'Age', 'Breed2'])['Fee'].transform('count') / alldata.shape[0]
    
    alldata['#Feature_state_type_breed1_age_freq']     = alldata[['Type', 'State', 'Breed1', 'Age']].groupby(['Type', 'State', 'Breed1'])['Age'].transform('mean')
    alldata['#Feature_state_type_breed1_age_fee_freq'] = alldata[['Type', 'State', 'Breed1', 'Age', 'Fee']].groupby(['Type', 'State', 'Breed1', 'Age'])['Fee'].transform('mean')
    alldata['#Feature_state_type_breed2_age_freq']     = alldata[['Type', 'State', 'Breed2', 'Age']].groupby(['Type', 'State', 'Breed2'])['Age'].transform('mean')
    alldata['#Feature_state_type_breed2_age_fee_freq'] = alldata[['Type', 'State', 'Breed2', 'Age', 'Fee']].groupby(['Type', 'State', 'Breed2', 'Age'])['Fee'].transform('mean')
    
    ###########################################################################################################
    
    alldata['RelAge'] = alldata[['Type', 'Age']].apply(relative_age, axis=1)
    alldata['IsNameGenuine'] = alldata[['Name', 'Quantity']].apply(genuine_name, axis=1)
    alldata['InstaFeature'] = alldata[['PhotoAmt', 'VideoAmt']].apply(seo_value, axis=1)
    alldata['ShowsMore'] = alldata['PhotoAmt'].apply(VerifibalePhotoAmy)
    alldata['age_log'] = np.log1p(alldata['Age'])
    
    alldata["Vaccinated_Deworked_Mutation"] = alldata['Vaccinated'].apply(str) + "_" + alldata['Dewormed'].apply(str)
    alldata = pd.get_dummies(alldata, columns=['Vaccinated_Deworked_Mutation'], prefix="Vaccinated_Dewormed")
    
    alldata["Vaccinated_Sterilized_Mutation"] = alldata['Vaccinated'].apply(str) + "_" + alldata['Sterilized'].apply(str)
    alldata = pd.get_dummies(alldata, columns=['Vaccinated_Sterilized_Mutation'], prefix="Vaccinated_Sterilized")
    
    alldata["Vaccinated_Health_Mutation"] = alldata['Vaccinated'].apply(str) + "_" + alldata['Health'].apply(str)
    alldata = pd.get_dummies(alldata, columns=['Vaccinated_Health_Mutation'], prefix="Vaccinated_Health")
    
    alldata["Sterilized_Dewormed_Mutation"] = alldata['Sterilized'].apply(str) + "_" + alldata['Dewormed'].apply(str)
    alldata = pd.get_dummies(alldata, columns=['Sterilized_Dewormed_Mutation'], prefix="Sterilized_Dewormed")
    
    alldata["Sterilized_Health_Mutation"] = alldata['Sterilized'].apply(str) + "_" + alldata['Health'].apply(str)
    alldata = pd.get_dummies(alldata, columns=['Sterilized_Health_Mutation'], prefix="Sterilized_Health")
    
    alldata['GlobalInstaRank'] = alldata['InstaFeature'].rank(method='max')
    print(">> Ranking Features By State")
    alldata = rankbyG(alldata, "State")
    print(">> Ranking Features By Animal")
    alldata = rankbyG(alldata, "Type")
    print(">> Ranking Features By Breed1")
    alldata = rankbyG(alldata, "Breed1")
    print(">> Ranking Features By Gender")
    alldata = rankbyG(alldata, "Gender")

    top_dogs = [179, 205, 195, 178, 206, 109, 189, 103]
    top_cats = [276, 268, 285, 252, 243, 251, 288, 247, 280, 290]

    alldata['#Feature_SecondaryColors'] = alldata['Color2'] + alldata['Color3']
    alldata['#Feature_MonoColor'] = np.where(alldata['#Feature_SecondaryColors'], 1, 0)
    alldata['top_breeds'] = 0
    alldata.loc[alldata['Breed1'].isin(top_dogs + top_cats) == True, 'top_breeds'] = 1
    alldata['top_breed_free'] = 0
    alldata.loc[alldata[(alldata['Fee'] == 0) & (alldata['top_breeds'] == 1)].index, 'top_breed_free'] = 1
    alldata['free_pet'] = 0
    alldata.loc[alldata[alldata['Fee'] == 0].index, 'free_pet'] = 1
    alldata['free_pet_age_1'] = 0
    alldata.loc[alldata[(alldata['Fee'] == 0) & (alldata['Age'] == 1)].index, 'free_pet_age_1'] = 1
    alldata['year'] = alldata['Age'] / 12.
    alldata['#Feature_less_a_year'] = np.where(alldata['Age'] < 12, 1, 0)
    alldata['#Feature_top_2_states'] = 0
    alldata.loc[alldata['State'].isin([41326, 41401]) == True, '#Feature_top_2_states'] = 1
    alldata['#Feature_age_exact'] = 0
    alldata.loc[alldata['Age'].isin([12, 24, 36, 48, 60, 72, 84, 96, 108]) == True, '#Feature_age_exact'] = 1
    alldata['#Feature_isLonely'] = np.where(alldata['Quantity'] > 1, 1, 0)
    alldata['total_img_video'] = alldata['PhotoAmt'] + alldata['VideoAmt']

    print("Mapping Breed Labels...")
    
    breed_label_map = {}
    for idx, row in (enumerate(labels_breed[['BreedID', 'BreedName']].values)):
        breed_label_map[row[1]] = int(row[0])
    
    dog_chars = pd.read_csv('../input/pet-breed-characteristics/dog_breed_characteristics.csv', usecols=['BreedName','MaleWtKg','AvgPupPrice', 'Intelligence',
    'Watchdog','PopularityUS2017', 'Temperment'])
    
    dog_chars['Temperment'] = dog_chars['Temperment'].astype(str).apply(lambda x: len(x.split(',')))
    dog_chars['Temperment'] = dog_chars['Temperment'].astype(int)
    dog_chars['BreedName'] = dog_chars['BreedName'].map(breed_label_map)
    dog_chars = dog_chars.add_prefix('dogs_')
    alldata = alldata.merge(dog_chars, how='left', left_on='Breed1', right_on='dogs_BreedName')
    alldata.drop('dogs_BreedName', axis=1, inplace=True)
    
    dog_chars = pd.read_csv('../input/pet-breed-characteristics/cat_breed_characteristics.csv', usecols=['BreedName','LapCat','Fur','MaleWtKg','AvgKittenPrice',\
    'PopularityUS2017', 'Temperment'])
    lap_map = {'Lap':3,'Non Lap':2, 'Generic':4, 'Rodent':1}
    dog_chars['LapCat'] = dog_chars['LapCat'].map(lap_map)
    fur_map = {'Medium':3,'Short':2, 'Long':4, 'Bald':1}
    dog_chars['Fur'] = dog_chars['Fur'].map(fur_map)
    del lap_map, fur_map
    dog_chars['Temperment'] = dog_chars['Temperment'].astype(str).apply(lambda x: len(x.split(',')))
    dog_chars['Temperment'] = dog_chars['Temperment'].astype(int)
    dog_chars['BreedName'] = dog_chars['BreedName'].map(breed_label_map)
    dog_chars = dog_chars.add_prefix('cats_')
    alldata = alldata.merge(dog_chars, how='left', left_on='Breed1', right_on='cats_BreedName')
    alldata.drop('cats_BreedName', axis=1, inplace=True)
    del dog_chars
    
    rescuer_count = alldata.groupby(['RescuerID'])['PetID'].count().reset_index()
    rescuer_count.columns = ['RescuerID', 'RescuerID_COUNT']
    alldata = alldata.merge(rescuer_count, how='left', on='RescuerID')

    Description_count = alldata.groupby(['Description'])['PetID'].count().reset_index()
    Description_count.columns = ['Description', 'Description_COUNT']
    alldata = alldata.merge(Description_count, how='left', on='Description')

    Name_count = alldata.groupby(['Name'])['PetID'].count().reset_index()
    Name_count.columns = ['Name', 'Name_COUNT']
    alldata = alldata.merge(Name_count, how='left', on='Name')

    agg = {}
    agg['Quantity'] = ['mean', 'max', 'min', 'skew', 'median']
    agg['Fee'] = ['mean', 'max', 'min', 'skew', 'median']
    agg['Age'] = ['mean', 'sum', 'max', 'min', 'skew', 'median']
    agg['Breed1'] = ['nunique', 'max', 'min', 'skew', 'median']
    agg['Breed2'] = ['nunique', 'max', 'min', 'skew', 'median']
    agg['Type'] = ['nunique', 'max', 'min', 'skew', 'median']
    agg['Gender'] = ['nunique', 'max', 'min', 'skew', 'median']
    agg['Color1'] = ['nunique', 'max', 'min', 'skew', 'median']
    agg['Color2'] = ['nunique', 'max', 'min', 'skew', 'median']
    agg['Color3'] = ['nunique', 'max', 'min', 'skew', 'median']
    agg['MaturitySize'] = ['nunique', 'max', 'min', 'skew', 'median']
    agg['FurLength'] = ['nunique', 'max', 'min', 'skew', 'median']
    agg['Vaccinated'] = ['nunique', 'max', 'min', 'skew', 'median']
    agg['Sterilized'] = ['nunique', 'var', 'max', 'min', 'skew', 'median']
    agg['Health'] = ['nunique', 'var', 'max', 'min', 'skew', 'median']
    agg["PhotoAmt"] = ['nunique', 'var', 'max', 'min', 'skew', 'median']
    agg["RelAge"] = ['nunique', 'var', 'max', 'min', 'skew', 'median']
    agg['sentiment_document_score'] = ['mean', 'var', 'std', 'min', 'max']

    # RescuerID
    grouby = 'RescuerID'
    agg_df = agg_features(alldata, grouby, agg, grouby)
    alldata = alldata.merge(agg_df, on=grouby, how='left')
    
    agg_kurt_df = alldata.groupby(grouby)[list(agg.keys())].apply(pd.DataFrame.kurt)
    agg_kurt_df.columns = [f"{key}_kurt" for key in list(agg.keys())]
    alldata = alldata.merge(agg_kurt_df, on=grouby, how='left')
    
    agg_perc_df = alldata.groupby(grouby)[list(agg.keys())].quantile(.25)
    agg_perc_df.columns = [f"{key}_perc_25" for key in list(agg.keys())]
    alldata = alldata.merge(agg_perc_df, on=grouby, how='left')
    
    agg_perc_df = alldata.groupby(grouby)[list(agg.keys())].quantile(.75)
    agg_perc_df.columns = [f"{key}_perc_75" for key in list(agg.keys())]
    alldata = alldata.merge(agg_perc_df, on=grouby, how='left')

    train = alldata[:len(train)]
    test  = alldata[len(train):]

    return train, test

def image_dim_features(train, test):
    # Load IDs and Image data
    # ===========================================
    split_char = "/"
    train_df_ids = train[['PetID']]
    test_df_ids = test[['PetID']]

    train_image_files = glob.glob('../input/petfinder-adoption-prediction/train_images/*.jpg')
    test_image_files = glob.glob('../input/petfinder-adoption-prediction/test_images/*.jpg')

    train_df_imgs = pd.DataFrame(train_image_files)
    train_df_imgs.columns = ['image_filename']
    train_imgs_pets = train_df_imgs['image_filename'].apply(lambda x: x.split('/')[-1].split('-')[0])
    train_df_imgs = train_df_imgs.assign(PetID=train_imgs_pets)

    test_df_imgs = pd.DataFrame(test_image_files)
    test_df_imgs.columns = ['image_filename']
    test_imgs_pets = test_df_imgs['image_filename'].apply(lambda x: x.split('/')[-1].split('-')[0])
    test_df_imgs = test_df_imgs.assign(PetID=test_imgs_pets)

    # ===========================================

    train_df_imgs['image_size'] = train_df_imgs['image_filename'].apply(getSize)
    train_df_imgs['temp_size'] = train_df_imgs['image_filename'].apply(getDimensions)
    train_df_imgs['width'] = train_df_imgs['temp_size'].apply(lambda x: x[0])
    train_df_imgs['height'] = train_df_imgs['temp_size'].apply(lambda x: x[1])
    train_df_imgs = train_df_imgs.drop(['temp_size'], axis=1)

    test_df_imgs['image_size'] = test_df_imgs['image_filename'].apply(getSize)
    test_df_imgs['temp_size'] = test_df_imgs['image_filename'].apply(getDimensions)
    test_df_imgs['width'] = test_df_imgs['temp_size'].apply(lambda x: x[0])
    test_df_imgs['height'] = test_df_imgs['temp_size'].apply(lambda x: x[1])
    test_df_imgs = test_df_imgs.drop(['temp_size'], axis=1)

    aggs = {
        'image_size': ['sum', 'mean', 'var'],
        'width': ['sum', 'mean', 'var', 'min', 'max'],
        'height': ['sum', 'mean', 'var', 'min', 'max'],
    }

    agg_train_imgs = train_df_imgs.groupby('PetID').agg(aggs)
    new_columns = [k + '_' + agg for k in aggs.keys() for agg in aggs[k]]
    agg_train_imgs.columns = new_columns
    agg_train_imgs = agg_train_imgs.reset_index()

    agg_test_imgs = test_df_imgs.groupby('PetID').agg(aggs)
    new_columns = [k + '_' + agg for k in aggs.keys() for agg in aggs[k]]
    agg_test_imgs.columns = new_columns
    agg_test_imgs = agg_test_imgs.reset_index()

    agg_imgs = pd.concat([agg_train_imgs, agg_test_imgs], axis=0).reset_index(drop=True)
    
    return agg_imgs

def metadata_features(train, test):

    if not preload:
        
        train_pet_ids = train.PetID.unique()
        test_pet_ids = test.PetID.unique()

        # Train Feature Extractions
        # ===============================

        dfs_train = Parallel(n_jobs=4, verbose=1)(delayed(extract_additional_features)(i, mode='train') for i in train_pet_ids)
        train_dfs_sentiment = [x[0] for x in dfs_train if isinstance(x[0], pd.DataFrame)]
        train_dfs_metadata = [x[1] for x in dfs_train if isinstance(x[1], pd.DataFrame)]
        train_dfs_sentiment = pd.concat(train_dfs_sentiment, ignore_index=True, sort=False)
        train_dfs_metadata = pd.concat(train_dfs_metadata, ignore_index=True, sort=False)

        # Test Feature Extractions
        # ===============================
        dfs_test = Parallel(n_jobs=4, verbose=1)(delayed(extract_additional_features)(i, mode='test') for i in test_pet_ids)
        test_dfs_sentiment = [x[0] for x in dfs_test if isinstance(x[0], pd.DataFrame)]
        test_dfs_metadata = [x[1] for x in dfs_test if isinstance(x[1], pd.DataFrame)]
        test_dfs_sentiment = pd.concat(test_dfs_sentiment, ignore_index=True, sort=False)
        test_dfs_metadata = pd.concat(test_dfs_metadata, ignore_index=True, sort=False)

    else:
        
        train_dfs_sentiment = pd.read_csv("./processed_data/train_dfs_sentiment.csv")
        train_dfs_metadata = pd.read_csv("./processed_data/train_dfs_metadata.csv")
        test_dfs_sentiment = pd.read_csv("./processed_data/test_dfs_sentiment.csv")
        test_dfs_metadata = pd.read_csv("./processed_data/test_dfs_metadata.csv")

        train_dfs_sentiment['sentiment_entities'].fillna('', inplace=True)
        train_dfs_metadata['metadata_annots_top_desc'].fillna('', inplace=True)
        test_dfs_sentiment['sentiment_entities'].fillna('', inplace=True)
        test_dfs_metadata['metadata_annots_top_desc'].fillna('', inplace=True)

    # Meta data Aggregates
    # ===============================
    aggregates = ['mean', 'sum', 'var']

    # Train Aggregates
    # ---------------------------
    train_metadata_desc = train_dfs_metadata.groupby(['PetID'])['metadata_annots_top_desc'].unique()
    train_metadata_desc = train_metadata_desc.reset_index()
    train_metadata_desc['metadata_annots_top_desc'] = train_metadata_desc['metadata_annots_top_desc'].apply(
        lambda x: ' '.join(x.tolist()))

    prefix = 'metadata'
    train_metadata_gr = train_dfs_metadata.drop(['metadata_annots_top_desc', 'metadata_text_annots_desc'], axis=1)
    for i in train_metadata_gr.columns:
        if 'PetID' not in i:
            train_metadata_gr[i] = train_metadata_gr[i].astype(float)
    train_metadata_gr = train_metadata_gr.groupby(['PetID']).agg(aggregates)
    train_metadata_gr.columns = pd.Index(['{}_{}_{}'.format(
        prefix, c[0], c[1].upper()) for c in train_metadata_gr.columns.tolist()])
    train_metadata_gr = train_metadata_gr.reset_index()

    train_sentiment_desc = train_dfs_sentiment.groupby(['PetID'])['sentiment_entities'].unique()
    train_sentiment_desc = train_sentiment_desc.reset_index()
    train_sentiment_desc['sentiment_entities'] = train_sentiment_desc['sentiment_entities'].apply(lambda x: ' '.join(x.tolist()))

    prefix = 'sentiment'
    train_sentiment_gr = train_dfs_sentiment.drop(['sentiment_entities'], axis=1)
    for i in train_sentiment_gr.columns:
        if 'PetID' not in i:
            train_sentiment_gr[i] = train_sentiment_gr[i].astype(float)
    train_sentiment_gr = train_sentiment_gr.groupby(['PetID']).agg(aggregates)
    train_sentiment_gr.columns = pd.Index(['{}_{}_{}'.format(
        prefix, c[0], c[1].upper()) for c in train_sentiment_gr.columns.tolist()])
    train_sentiment_gr = train_sentiment_gr.reset_index()

    # Test data Aggregates
    # ---------------------------
    test_metadata_desc = test_dfs_metadata.groupby(['PetID'])['metadata_annots_top_desc'].unique()
    test_metadata_desc = test_metadata_desc.reset_index()
    test_metadata_desc['metadata_annots_top_desc'] = test_metadata_desc['metadata_annots_top_desc'].apply(lambda x: ' '.join(x.tolist()))

    prefix = 'metadata'
    test_metadata_gr = test_dfs_metadata.drop(['metadata_annots_top_desc', 'metadata_text_annots_desc'], axis=1)
    for i in test_metadata_gr.columns:
        if 'PetID' not in i:
            test_metadata_gr[i] = test_metadata_gr[i].astype(float)
    test_metadata_gr = test_metadata_gr.groupby(['PetID']).agg(aggregates)
    test_metadata_gr.columns = pd.Index(['{}_{}_{}'.format(
        prefix, c[0], c[1].upper()) for c in test_metadata_gr.columns.tolist()])
    test_metadata_gr = test_metadata_gr.reset_index()

    test_sentiment_desc = test_dfs_sentiment.groupby(['PetID'])['sentiment_entities'].unique()
    test_sentiment_desc = test_sentiment_desc.reset_index()
    test_sentiment_desc['sentiment_entities'] = test_sentiment_desc['sentiment_entities'].apply(lambda x: ' '.join(x.tolist()))

    prefix = 'sentiment'
    test_sentiment_gr = test_dfs_sentiment.drop(['sentiment_entities'], axis=1)
    for i in test_sentiment_gr.columns:
        if 'PetID' not in i:
            test_sentiment_gr[i] = test_sentiment_gr[i].astype(float)
    test_sentiment_gr = test_sentiment_gr.groupby(['PetID']).agg(aggregates)
    test_sentiment_gr.columns = pd.Index(['{}_{}_{}'.format(
        prefix, c[0], c[1].upper()) for c in test_sentiment_gr.columns.tolist()])
    test_sentiment_gr = test_sentiment_gr.reset_index()
    
    
    ## DROP THESE COLS AS THEY ARE USELESS WARNING MAKE SURE YOUUPDTE THIS SECTION ALWAYS IF YOU ADD"REMOVE ANY
    drop_cols = [
        'sentiment_sentiment_magnitude_MEAN', 'sentiment_sentiment_magnitude_VAR',
        'sentiment_sentiment_score_MEAN', 'sentiment_sentiment_score_VAR',
        'sentiment_sentiment_document_magnitude_MEAN', 'sentiment_sentiment_document_magnitude_VAR',
        'sentiment_sentiment_document_score_MEAN', 'sentiment_sentiment_document_score_VAR',
        'sentiment_sentiment_salience_mean_MEAN', 'sentiment_sentiment_salience_mean_VAR',
        'sentiment_sentiment_salience_max_MEAN', 'sentiment_sentiment_salience_max_VAR',
        'sentiment_sentiment_salience_min_MEAN', 'sentiment_sentiment_salience_min_VAR',
        'sentiment_sentiment_salience_len_MEAN', 'sentiment_sentiment_salience_len_VAR',
        'sentiment_sentiment_mentions_mean_length_MEAN', 'sentiment_sentiment_mentions_mean_length_VAR',
        'sentiment_sentiment_mentions_len_MEAN', 'sentiment_sentiment_mentions_len_VAR',
        'sentiment_sentiment_mentions_total_MEAN', 'sentiment_sentiment_mentions_total_VAR',
        ]
    train_sentiment_gr.drop(drop_cols, axis=1, inplace=True)
    test_sentiment_gr.drop(drop_cols, axis=1, inplace=True)
    
    # Mergining Features with Train/Test
    # =======================================
    train_proc = train.copy()
    train_proc = train_proc.merge(train_sentiment_gr, how='left', on='PetID')
    train_proc = train_proc.merge(train_metadata_gr, how='left', on='PetID')
    train_proc = train_proc.merge(train_metadata_desc, how='left', on='PetID')
    train_proc = train_proc.merge(train_sentiment_desc, how='left', on='PetID')

    test_proc = test.copy()
    test_proc = test_proc.merge(test_sentiment_gr, how='left', on='PetID')
    test_proc = test_proc.merge(test_metadata_gr, how='left', on='PetID')
    test_proc = test_proc.merge(test_metadata_desc, how='left', on='PetID')
    test_proc = test_proc.merge(test_sentiment_desc, how='left', on='PetID')
    
    return train_proc, test_proc

def breed_maps(train_proc, test_proc, labels_breed):
    
    train_breed_main = train_proc[['Breed1']].merge(labels_breed, how='left', left_on='Breed1', right_on='BreedID',
                                                    suffixes=('', '_main_breed'))
    train_breed_main = train_breed_main.iloc[:, 2:]
    train_breed_main = train_breed_main.add_prefix('main_breed_')
    train_breed_second = train_proc[['Breed2']].merge(labels_breed, how='left', left_on='Breed2', right_on='BreedID',
                                                      suffixes=('', '_second_breed'))
    train_breed_second = train_breed_second.iloc[:, 2:]
    train_breed_second = train_breed_second.add_prefix('second_breed_')
    train_proc = pd.concat([train_proc, train_breed_main, train_breed_second], axis=1)
    test_breed_main = test_proc[['Breed1']].merge(labels_breed, how='left', left_on='Breed1', right_on='BreedID',
                                                  suffixes=('', '_main_breed'))
    test_breed_main = test_breed_main.iloc[:, 2:]
    test_breed_main = test_breed_main.add_prefix('main_breed_')
    test_breed_second = test_proc[['Breed2']].merge(labels_breed, how='left', left_on='Breed2', right_on='BreedID',
                                                    suffixes=('', '_second_breed'))
    test_breed_second = test_breed_second.iloc[:, 2:]
    test_breed_second = test_breed_second.add_prefix('second_breed_')
    test_proc = pd.concat([test_proc, test_breed_main, test_breed_second], axis=1)
    
    X = pd.concat([train_proc, test_proc], ignore_index=True, sort=False)
    categorical_columns = ['main_breed_BreedName', 'second_breed_BreedName', 'group_pets'] #new addition group_pets
    
    for i in categorical_columns:
        X.loc[:, i] = pd.factorize(X.loc[:, i])[0]
    return X

def nlp_features(X_temp):
    
    from sklearn.feature_extraction.text import CountVectorizer
    
    text_columns = ['Description', 'metadata_annots_top_desc', 'sentiment_entities']
    X_text = X_temp[text_columns]

    for i in X_text.columns:
        X_text.loc[:, i] = X_text.loc[:, i].fillna('<MISSING>')

    n_components = 50
    text_features = []

    # Generate text features:
    for i in X_text.columns:
        # Initialize decomposition methods:
        print('Generating features from: {}'.format(i))
        svd_ = TruncatedSVD(n_components=n_components, random_state=1337)
        nmf_ = NMF(n_components=n_components, random_state=1337)
        
        tfidf_col = TfidfVectorizer().fit_transform(X_text.loc[:, i].values)
        svd_col = svd_.fit_transform(tfidf_col)
        svd_col = pd.DataFrame(svd_col)
        svd_col = svd_col.add_prefix('SVD_{}_'.format(i))
        nmf_col = nmf_.fit_transform(tfidf_col)
        nmf_col = pd.DataFrame(nmf_col)
        nmf_col = nmf_col.add_prefix('NMF_{}_'.format(i))
        text_features.append(svd_col)
        text_features.append(nmf_col)

    # Combine all extracted features:
    text_features = pd.concat(text_features, axis=1)
    # Concatenate with main DF:
    X_temp = pd.concat([X_temp, text_features], axis=1)
    # Remove raw text columns:
    for i in X_text.columns:
        X_temp = X_temp.drop(i, axis=1)
    # Remove unnecessary columns:
    to_drop_columns = ['PetID', 'Name']
    X_temp = X_temp.drop(to_drop_columns, axis=1)

    return X_temp

def run_lgbm(X_temp, test):
    
    params = {
        'application': 'regression',
        'boosting': 'gbdt',
        'metric': 'rmse',
        'num_leaves': 35,
        'max_depth': 9,
        'learning_rate': 0.01, #can't chnage this
        'bagging_fraction': 0.512,  # .85 previously
        'feature_fraction': 0.5177,  # .8 previously
        'min_split_gain': 0.0845,
        'min_child_samples': 24,
        'min_child_weight': 0.036,
        'lambda_l1': 5.0334,
        'lambda_l2': 7.250,
        'verbosity': -1,
        'data_random_seed': 1337,
        }
    
    # Additional parameters:
    early_stop = 500
    verbose_eval = 500
    num_rounds = 4000 #8000
    n_splits = 10 #10
    
    print(f'Params For LGB are \n {params}')
    
    # Split into train and test again:
    X_train = X_temp.loc[np.isfinite(X_temp.AdoptionSpeed), :]
    X_test = X_temp.loc[~np.isfinite(X_temp.AdoptionSpeed), :]

    # Remove missing target column from test:
    X_test = X_test.drop(['AdoptionSpeed'], axis=1)

    print('X_train shape: {}'.format(X_train.shape))
    print('X_test shape: {}'.format(X_test.shape))

    # Check if columns between the two DFs are the same:
    train_cols = X_train.columns.tolist()
    train_cols.remove('AdoptionSpeed')
    train_cols.remove('RescuerID')

    test_cols = X_test.columns.tolist()

    kfold = StratifiedKFold(n_splits=n_splits, random_state=1337)
    oof_train = np.zeros((X_train.shape[0]))
    oof_test = np.zeros((X_test.shape[0], n_splits))

    rescuer_gb_mean = X_train.groupby('RescuerID')['AdoptionSpeed'].agg("mean").reset_index()
    rescuer_gb_mean.columns = ['RescuerID', 'AdoptionSpeed_mean']

    rescuer_ids = rescuer_gb_mean['RescuerID'].values
    rescuer_as_mean = rescuer_gb_mean['AdoptionSpeed_mean'].values

    i, cv_scores = 0, []
    
    for train_index, valid_index in kfold.split(rescuer_ids, rescuer_as_mean.astype(np.int)):
        
        rescuser_train_ids = rescuer_ids[train_index]
        rescuser_valid_ids = rescuer_ids[valid_index]

        X_tr = X_train[X_train["RescuerID"].isin(rescuser_train_ids)]
        X_val = X_train[X_train["RescuerID"].isin(rescuser_valid_ids)]
    
        y_tr = X_tr['AdoptionSpeed'].values
        X_tr = X_tr.drop(['AdoptionSpeed', 'RescuerID'], axis=1)

        y_val = X_val['AdoptionSpeed'].values
        X_val = X_val.drop(['AdoptionSpeed', 'RescuerID'], axis=1)

        print('\ny_tr distribution: {}'.format(Counter(y_tr)))

        d_train = lgb.Dataset(X_tr, label=y_tr)
        d_valid = lgb.Dataset(X_val, label=y_val)
        watchlist = [d_train, d_valid]

        print(f'Training LGB For FOLD {i+1}:')
        model = lgb.train(params,
                          train_set=d_train,
                          num_boost_round=num_rounds,
                          valid_sets=watchlist,
                          verbose_eval=verbose_eval,
                          early_stopping_rounds=early_stop
                          )

        val_pred = model.predict(X_val, num_iteration=model.best_iteration)
        test_pred = model.predict(X_test.drop(["RescuerID"], axis=1), num_iteration=model.best_iteration)

        oof_train[X_val.index] = val_pred
        oof_test[:, i] = test_pred

        i += 1
        
        optR = OptimizedRounder()
        optR.fit(val_pred, y_val)
        coefficients__ = optR.coefficients()
        coefficients__[0] = 1.645
        #change this only leads to better score....
        pred_test_y_k_ = optR.predict(val_pred, coefficients__)
        print("Predicted Counts = ", Counter(pred_test_y_k_))
        print("Coefficients = ", coefficients__)
        qwk = quadratic_weighted_kappa(y_val, pred_test_y_k_)
        print(f'Fold {i} QWK = ', qwk)
        print('#'*35)
        #cache them...
        cv_scores.append(qwk)
    
    #print stats
    print('All Folds QWK LGBM', cv_scores)
    print('CV STD LGBM', np.std(cv_scores), 'CV MEAN QWK', np.mean(cv_scores))
    
    ### imp df
    imp_df = pd.DataFrame()
    imp_df["feature"] = list(train_cols)
    imp_df["importance_gain"] = model.feature_importance(importance_type='gain')
    imp_df["importance_split"] = model.feature_importance(importance_type='split')
    imp_df.to_csv('imps.csv', index=False)

    # Compute QWK based on OOF train predictions:
    optR = OptimizedRounder()
    optR.fit(oof_train, X_train['AdoptionSpeed'].values)
    coefficients = optR.coefficients()
    pred_test_y_k = optR.predict(oof_train, coefficients)
    print("\nValid Counts = ", Counter(X_train['AdoptionSpeed'].values))
    print("Predicted Counts = ", Counter(pred_test_y_k))
    print("Coefficients = ", coefficients)
    qwk = quadratic_weighted_kappa(X_train['AdoptionSpeed'].values, pred_test_y_k)
    print("Final QWK = ", qwk)

    coefficients_ = coefficients.copy()
    print(f'coefficients returned From optim for LGBM are {coefficients_}')

    coefficients_[0] = 1.645
    #coefficients_[1] = 2.115
    #coefficients_[3] = 2.84

    print(f'coefficients actually used are {coefficients_}')

    train_predictions = optR.predict(oof_train, coefficients_).astype(int)
    print('train pred distribution: {}'.format(Counter(train_predictions)))

    test_predictions = optR.predict(oof_test.mean(axis=1), coefficients_)
    print('test pred distribution: {}'.format(Counter(test_predictions)))

    # Distribution inspection of original target and predicted train and test:
    print("True Distribution:")
    print(pd.value_counts(X_train['AdoptionSpeed'], normalize=True).sort_index())
    print("\nTrain Predicted Distribution:")
    print(pd.value_counts(train_predictions, normalize=True).sort_index())
    print("\nTest Predicted Distribution:")
    print(pd.value_counts(test_predictions, normalize=True).sort_index())
    submission = pd.DataFrame({'PetID': test['PetID'].values, 'AdoptionSpeed': test_predictions.astype(np.uint16)})
    
    try:
        from string import ascii_uppercase
        from pandas import DataFrame
        import seaborn as sn
        from sklearn.metrics import confusion_matrix, accuracy_score

        columns = ['class %s' %(i) for i in list(ascii_uppercase)[0:len(np.unique(train_predictions))]]
        confm = confusion_matrix(X_train['AdoptionSpeed'].values, train_predictions)
        df_cm = DataFrame(confm, index=columns, columns=columns)
        print(accuracy_score(X_train['AdoptionSpeed'].values, train_predictions))
        ax = sn.heatmap(df_cm, cmap='Oranges', annot=True, fmt="d" )
        plt.savefig('cnf_mtx_LGBM.png')
        plt.show()
    except:
        print('Couldnt Run The CNFMTRX...')

    return submission, oof_train, oof_test

def run_xgb(X_temp, test, params):
              
    n_splits = 10
    verbose_eval = 100
    num_rounds = 4000 #8000
    early_stop = 500
    
    print(f'Params For XGB are \n {params}')
    # Split into train and test again:
    X_train = X_temp.loc[np.isfinite(X_temp.AdoptionSpeed), :]
    X_test = X_temp.loc[~np.isfinite(X_temp.AdoptionSpeed), :]

    # Remove missing target column from test:
    X_test = X_test.drop(['AdoptionSpeed', "RescuerID"], axis=1)

    print('X_train shape: {}'.format(X_train.shape))
    print('X_test shape: {}'.format(X_test.shape))

    # Check if columns between the two DFs are the same:
    train_cols = X_train.columns.tolist()
    train_cols.remove('AdoptionSpeed')
    train_cols.remove('RescuerID')

    test_cols = X_test.columns.tolist()

    kfold = StratifiedKFold(n_splits=n_splits, random_state=1337)
    oof_train = np.zeros((X_train.shape[0]))
    oof_test = np.zeros((X_test.shape[0], n_splits))

    rescuer_gb_mean = X_train.groupby('RescuerID')['AdoptionSpeed'].agg("mean").reset_index()
    rescuer_gb_mean.columns = ['RescuerID', 'AdoptionSpeed_mean']

    rescuer_ids = rescuer_gb_mean['RescuerID'].values
    rescuer_as_mean = rescuer_gb_mean['AdoptionSpeed_mean'].values

    i, cv_scores = 0, []

    for train_index, valid_index in kfold.split(X_train, X_train['AdoptionSpeed'].astype(np.int)):
        
        print(f'Fold {i+1}')
        try:
            os.system('rm core')
            print('Core file Error')
        except:
            print('No temp file Core Generated...')
        
        X_tr = X_train.iloc[train_index]
        X_val = X_train.iloc[valid_index]

        y_tr = X_tr['AdoptionSpeed'].values
        X_tr = X_tr.drop(['AdoptionSpeed', 'RescuerID'], axis=1)

        y_val = X_val['AdoptionSpeed'].values
        X_val = X_val.drop(['AdoptionSpeed', 'RescuerID'], axis=1)

        d_train = xgb.DMatrix(data=X_tr, label=y_tr, feature_names=X_tr.columns)
        d_valid = xgb.DMatrix(data=X_val, label=y_val, feature_names=X_val.columns)

        watchlist = [(d_train, 'train'), (d_valid, 'valid')]
        model = xgb.train(dtrain=d_train, num_boost_round=num_rounds, evals=watchlist,early_stopping_rounds=early_stop, verbose_eval=verbose_eval, params=params)
        
        del d_train, d_valid
        gc.collect()
        
        valid_pred = model.predict(xgb.DMatrix(X_val, feature_names=X_val.columns), ntree_limit=model.best_ntree_limit)
        test_pred = model.predict(xgb.DMatrix(X_test, feature_names=X_test.columns), ntree_limit=model.best_ntree_limit)

        oof_train[X_val.index] = valid_pred
        oof_test[:, i] = test_pred

        i += 1
        
        optR = OptimizedRounder()
        optR.fit(valid_pred, y_val)
        coefficients__ = optR.coefficients()
        coefficients__[0] = 1.645
        #changing this only leads to better score....
        pred_test_y_k_ = optR.predict(valid_pred, coefficients__)
        print("Predicted Counts = ", Counter(pred_test_y_k_))
        print("Coefficients = ", coefficients__)
        qwk = quadratic_weighted_kappa(y_val, pred_test_y_k_)
        print(f'Fold {i} QWK = ', qwk)
        print('#'*35)
        #cache them...
        cv_scores.append(qwk)

    #print stats
    print('All Folds QWK XGB', cv_scores)
    print('CV STD XGB', np.std(cv_scores), 'CV MEAN QWK', np.mean(cv_scores))
    
    ##############################################
    optR = OptimizedRounder()
    optR.fit(oof_train, X_train['AdoptionSpeed'].values)
    coefficients = optR.coefficients()
    valid_pred = optR.predict(oof_train, coefficients)
    qwk = quadratic_weighted_kappa(X_train['AdoptionSpeed'].values, valid_pred)
    print("QWK = ", qwk)
    coefficients_ = coefficients.copy()
    print(f'coefficients returned From optim for XGB are {coefficients_}')
    coefficients_[0] = 1.645
    # coefficients_[1] = 2.115
    # coefficients_[3] = 2.84
    np.save('coeff_.npy', coefficients_) #cache it
    print(f'coefficients used for XGB are {coefficients_}')
    train_predictions = optR.predict(oof_train, coefficients_).astype(np.int8)
    print(f'train pred distribution: {Counter(train_predictions)}')
    test_predictions = optR.predict(oof_test.mean(axis=1), coefficients_).astype(np.int8)
    print(f'test pred distribution: {Counter(test_predictions)}')

    submission = pd.DataFrame({'PetID': test['PetID'].values, 'AdoptionSpeed': test_predictions})
    
    try:
        from string import ascii_uppercase
        from pandas import DataFrame
        import seaborn as sn
        from sklearn.metrics import confusion_matrix, accuracy_score

        columns = ['class %s' %(i) for i in list(ascii_uppercase)[0:len(np.unique(train_predictions))]]
        confm = confusion_matrix(X_train['AdoptionSpeed'].values, train_predictions)
        df_cm = DataFrame(confm, index=columns, columns=columns)
        print(accuracy_score(X_train['AdoptionSpeed'].values, train_predictions))
        plt.figure()
        ax_1 = sn.heatmap(df_cm, cmap='Oranges', annot=True, fmt="d" )
        plt.show()
        plt.savefig('cnf_mtx_XGB.png')
    except:
        print('Couldnt Run The CNFMTRX...')
        
    return submission, oof_train, oof_test

#mlcrate
def train_kfold(model, x_train, y_train, x_test=None, folds=5, metrics=None, predict_type='predict_proba', stratify=None, random_state=1337, skip_checks=False):

    from sklearn.model_selection import KFold, StratifiedKFold  # Optional dependencies
    from sklearn.base import clone
    import numpy as np
    
    x_train = x_train.drop(['AdoptionSpeed', "RescuerID"], axis=1)
    
    if hasattr(x_train, 'columns'):
        columns = x_train.columns.values
        columns_exists = True
    else:
        columns_exists = False
        
    x_train = x_train.fillna(-1, axis=1)
    x_test = x_test.fillna(-1, axis=1)
    
    x_train = x_train.fillna(method='ffill', axis=1)
    x_test  = x_test.fillna(method='ffill', axis=1)
    
    x_test = x_test.drop(['AdoptionSpeed', "RescuerID"], axis=1)
    
    x_train = np.asarray(x_train)
    y_train = np.array(y_train)

    if not hasattr(metrics, '__iter__'):
        metrics = [metrics]

    if x_test is not None:
        if columns_exists and not skip_checks:
            try:
                x_test = x_test[columns]
            except Exception as e:
                print('Could not coerce x_test columns to match x_train columns. Set skip_checks=True to run anyway.')
                raise e

        x_test = np.asarray(x_test)

    if not skip_checks and x_test is not None:
        assert x_train.shape[1] == x_test.shape[1], "x_train and x_test have different numbers of features."

    print('Training {} {}{} models on training set {} {}'.format(folds, 'stratified ' if stratify is not None else '', type(model),
            x_train.shape, 'with test set {}'.format(x_test.shape) if x_test is not None else 'without a test set'))

    # Init a timer to get fold durations
    t = Timer()

    if stratify is not None:
        kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
        splits = kf.split(x_train, stratify)
    else:
        kf = KFold(n_splits=folds, shuffle=True, random_state=4242)
        splits = kf.split(x_train)

    p_train = np.zeros_like(y_train, dtype=np.float32)
    ps_test = []
    models = []

    fold_i = 0
    for train_kf, valid_kf in splits:
        print('Running fold {}, {} train samples, {} validation samples'.format(fold_i, len(train_kf), len(valid_kf)))
        
        x_tr, y_tr = x_train[train_kf], y_train[train_kf]
        x_va, y_va = x_train[valid_kf], y_train[valid_kf]

        # Start a timer for the fold
        t.add('fold{}'.format(fold_i))

        mdl = clone(model)
        mdl.fit(x_tr, y_tr)

        p_va = mdl.predict(x_va)
        p_test = mdl.predict(x_test)

        p_train[valid_kf] = p_va

        print('Finished training fold {} - took {}'.format(fold_i, t.format_elapsed('fold{}'.format(fold_i))))

        ps_test.append(p_test)
        models.append(mdl)

        fold_i += 1

    if x_test is not None:
        p_test = np.mean(ps_test, axis=0)
    else:
        p_test = None

    print('Finished training {} models, took {}'.format(folds, t.format_elapsed(0)))

    return models, p_train, p_test

def load_image_nima(path):
    
    from keras.preprocessing.image import load_img, img_to_array
    from keras.applications.inception_resnet_v2 import preprocess_input
    
    img = load_img(path, target_size=(224,224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    return x

def build_model_nima(shape=(None, None, 3), weights_path="../input/titu1994neuralimageassessment/inception_resnet_weights.h5"):
    
    from keras.applications.inception_resnet_v2 import InceptionResNetV2
    
    base_model = InceptionResNetV2(input_shape= shape, include_top=False, pooling='avg', weights=None)
    x = Dropout(0.75)(base_model.output)
    x = Dense(10, activation='softmax')(x)

    model = Model(base_model.input, x)
    model.load_weights(weights_path)
    
    print('Done building NIA')
    return model

def mean_score(scores):
    si = np.arange(1, 11, 1)
    mean = np.sum(scores * si)
    return mean

def std_score(scores):
    si = np.arange(1, 11, 1)
    mean = mean_score(scores)
    std = np.sqrt(np.sum(((si - mean) ** 2) * scores))
    return std

def train_model_nima(model, train, test, nn_params={"batch_size": 128, "img_size": 224}):
    
    batch_size = nn_params['batch_size']
    img_size = nn_params['img_size']
    pet_ids = train['PetID'].values
    train_df_ids = train[['PetID']]
    n_batches = len(train_df_ids) // batch_size + 1

    # Train images NIMA
    features = pd.DataFrame(columns = ['PetID', 'NIMA_mean', 'NIMA_std'])
    train_image = glob.glob("../input/petfinder-adoption-prediction/train_images/*-1.jpg")
    print('Total Images TRAIN NIA', len(train_image))
    n_batches = len(train_image) // batch_size + 1
    for b in (range(n_batches)):
        start = b * batch_size
        end = (b + 1) * batch_size
        batch_pets = train_image[start:end]
        batch_images = np.zeros((len(batch_pets), img_size, img_size, 3))
        for i, pet_id in enumerate(batch_pets):
            try:
                batch_images[i] = load_image_nima(pet_id)
            except:
                pass
        batch_preds = model.predict(batch_images) #scores
        for i, pet_id in enumerate(batch_pets):
            mean = mean_score(batch_preds[i])
            std = std_score(batch_preds[i])
            features.loc[start+i] = [pet_id.split("/")[-1].split("-")[0], mean, std]
    
    train_feats = pd.DataFrame.from_dict(features)
    train_feats = train_feats.groupby('PetID').agg("mean").reset_index()
    del features
    gc.collect()
    
    
    #### TEST SIDE
    batch_size = nn_params['batch_size']
    img_size = nn_params['img_size']
    pet_ids = test['PetID'].values
    test_df_ids = test[['PetID']]
    n_batches = len(test_df_ids) // batch_size + 1

    # Test images NIMA
    features = pd.DataFrame(columns = ['PetID', 'NIMA_mean', 'NIMA_std'])
    test_image = glob.glob("../input/petfinder-adoption-prediction/test_images/*.jpg")
    print('Total Images in TEST', len(test_image))
    n_batches = len(test_image) // batch_size + 1
    for b in (range(n_batches)):
        start = b * batch_size
        end = (b + 1) * batch_size
        batch_pets = test_image[start:end]
        batch_images = np.zeros((len(batch_pets), img_size, img_size, 3))
        for i, pet_id in enumerate(batch_pets):
            try:
                batch_images[i] = load_image_nima(pet_id)
            except:
                pass
        batch_preds = model.predict(batch_images) #scores
        for i, pet_id in enumerate(batch_pets):
            mean = mean_score(batch_preds[i])
            std = std_score(batch_preds[i])
            features.loc[start+i] = [pet_id.split("/")[-1].split("-")[0], mean, std]
    
    test_feats = pd.DataFrame.from_dict(features)
    test_feats = test_feats.groupby('PetID').agg("mean").reset_index()
    del features
    
    #merge now
    train = pd.merge(train, train_feats, how='left', on=['PetID'])
    test  = pd.merge(test, test_feats, how='left', on=['PetID'])
    del train_feats, test_feats
    gc.collect()
    
    return train, test

# https://www.kaggle.com/ogrellier/python-target-encoding-for-categorical-features By Olivier Script...
def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))

def target_encode(trn_series=None, tst_series=None, target=None, min_samples_leaf=1, smoothing=1, noise_level=0):
    """
    Smoothing is computed like in the following paper by Daniele Micci-Barreca
    https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf
    trn_series : training categorical feature as a pd.Series
    tst_series : test categorical feature as a pd.Series
    target : target data as a pd.Series
    min_samples_leaf (int) : minimum samples to take category average into account
    smoothing (int) : smoothing effect to balance categorical average vs prior  
    """ 
    assert len(trn_series) == len(target)
    assert trn_series.name == tst_series.name
    temp = pd.concat([trn_series, target], axis=1)
    # Compute target mean 
    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])
    # Compute smoothing
    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
    # Apply average function to all target data
    prior = target.mean()
    # The bigger the count the less full_avg is taken into account
    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)
    # Apply averages to trn and tst series
    ft_trn_series = pd.merge(
        trn_series.to_frame(trn_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=trn_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_trn_series.index = trn_series.index 
    ft_tst_series = pd.merge(
        tst_series.to_frame(tst_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=tst_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_tst_series.index = tst_series.index
    return add_noise(ft_trn_series, noise_level), add_noise(ft_tst_series, noise_level)

def hu_moms(df, path = '../input/petfinder-adoption-prediction/train_images/*.jpg', is_train=True):
    
    from math import copysign, log10
    
    huMoments0=[]
    huMoments1=[]
    huMoments2=[]
    huMoments3=[]
    huMoments4=[]
    huMoments5=[]
    huMoments6=[]
    imageid =[]
    id_ = 'train'
    
    if is_train:
        image_info_train = glob.glob(path)
    else:
        id_ = 'test'
        image_info_train = glob.glob(path)
    
    for filename in image_info_train:
        
                if filename.endswith("-1.jpg"): # Take only the moments of picture 1
                
                    image = cv2.imread(filename)
                    im = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
                    # Calculate Moments
                    moments = cv2.moments(im)
                    # Calculate Hu Moments
                    huMoments = cv2.HuMoments(moments)
                    # Log scale hu moments
                    for i in range(0,7):
                          huMoments[i] = round(-1* copysign(1.0, huMoments[i]) * log10(abs(huMoments[i])),2)
    
                    #image id
                    imageid.append(filename.replace('.jpg','').replace(f'../input/petfinder-adoption-prediction/{id_}_images/', ''))
                    huMoments0.append(huMoments[0])
                    huMoments1.append(huMoments[1])
                    huMoments2.append(huMoments[2])
                    huMoments3.append(huMoments[3])
                    huMoments4.append(huMoments[4])
                    huMoments5.append(huMoments[5])
                    huMoments6.append(huMoments[6])
    
    image_moments_train = pd.concat([pd.DataFrame({'ImageId':imageid}),pd.DataFrame({'huMoments0':np.concatenate(huMoments0,axis=0)}), 
                                        pd.DataFrame({'huMoments1':np.concatenate(huMoments1,axis=0)}),
                                        pd.DataFrame({'huMoments2':np.concatenate(huMoments2,axis=0)}),
                                        pd.DataFrame({'huMoments3':np.concatenate(huMoments3,axis=0)}),
                                        pd.DataFrame({'huMoments4':np.concatenate(huMoments4,axis=0)}),
                                        pd.DataFrame({'huMoments5':np.concatenate(huMoments5,axis=0)}),
                                        pd.DataFrame({'huMoments6':np.concatenate(huMoments6,axis=0)})],
                                        axis=1)
            

    # create the PetId variable
    image_moments_train['PetID'] = image_moments_train['ImageId'].str.split('-').str[0]
    image_moments_train = image_moments_train[image_moments_train['ImageId'].apply(lambda x:x.endswith(("-1")))]
    image_moments_train = image_moments_train.drop(['ImageId'], 1)
    
    df = pd.merge(df, image_moments_train,  how='left', left_on=['PetID'], right_on = ['PetID'])
    del imageid, image_moments_train, huMoments6, huMoments5, huMoments4, huMoments3, huMoments2
    return df

###############################################################################################
###############################################################################################
############################ MAIN WORK STARTS HERE ############################################
###############################################################################################
###############################################################################################

# DON\'T CHANGE THE ORDERING AT ALL
print('Setting Seeds')
set_seed(2411)
#load datasets................................
train, test, labels_state, labels_breed, labels_color = load_tabular_data()

print('Generating Senti Stats')
train = add_sentimental_analysis(train, path_name= '../input/petfinder-adoption-prediction/train_sentiment/*.json', is_train = True)
test  = add_sentimental_analysis(test, path_name= '../input/petfinder-adoption-prediction/test_sentiment/*.json', is_train = False)

import fastai
from fastai.vision import *
print('fast.ai version:{}'.format(fastai.__version__))

print('Generating fes')
train, test = basic_features(train, test)

model = build_model_nima()
train, test = train_model_nima(model, train, test) #updated train and test with NIA feats as well
print(train.columns.tolist())
del model
gc.collect()

trn, sub = target_encode(train["Breed1"],test["Breed1"],target=train.AdoptionSpeed,min_samples_leaf=100,smoothing=10,noise_level=0.01)
train['tencode_breed1'] = trn
test['tencode_breed1'] = sub

trn, sub = target_encode(train["Breed2"],test["Breed2"],target=train.AdoptionSpeed,min_samples_leaf=100,smoothing=10,noise_level=0.01)
train['tencode_breed2'] = trn
test['tencode_breed2'] = sub

trn, sub = target_encode(train["Age"],test["Age"],target=train.AdoptionSpeed,min_samples_leaf=100,smoothing=10,noise_level=0.01)
train['tencode_Age'] = trn
test['tencode_Age'] = sub

del trn, sub
gc.collect()

print('Adding Image pixel vals')
train = add_image_fe(train, '../input/petfinder-adoption-prediction/train_images/*.jpg', is_train = True)
test  = add_image_fe(test,  '../input/petfinder-adoption-prediction/test_images/*.jpg', is_train = False)

# print('Adding HU MOMS')
# train = hu_moms(train, path = '../input/petfinder-adoption-prediction/train_images/*.jpg', is_train = True)
# test  = hu_moms(test,  path = '../input/petfinder-adoption-prediction/test_images/*.jpg', is_train = False)

print('Generating Feats NLP')
train =  meta_nlp_feats(train, 'Description')
test  =  meta_nlp_feats(test,  'Description')

print('Generating Bounding')
train = bounding_features(train, meta_path="../input/petfinder-adoption-prediction/train_metadata/")
test  = bounding_features(test, meta_path="../input/petfinder-adoption-prediction/test_metadata/")

# print('Generating Bounding Feats Across All files Called this func **bounding_features_all_files**')

# bounding_features_train = bounding_features_all_files(train, meta_path="../input/petfinder-adoption-prediction/train_metadata/*.json")
# bounding_features_test  = bounding_features_all_files(train, meta_path="../input/petfinder-adoption-prediction/test_metadata/*.json")
# bounding_features_train = bounding_features_train.groupby("PetID").agg(['sum', 'count','max','median','min'])
# bounding_features_test  = bounding_features_test.groupby("PetID").agg(['sum', 'count','max','median','min'])

# def flatten_cols(df):
#     df.columns = ['_'.join(tup).rstrip('_') for tup in df.columns.values]

# flatten_cols(bounding_features_train)
# bounding_features_train = bounding_features_train.reset_index()
# flatten_cols(bounding_features_test)
# bounding_features_test = bounding_features_test.reset_index()

# train = pd.merge(train, bounding_features_train, on='PetID', how='left')
# test  = pd.merge(test, bounding_features_test, on='PetID', how='left')

# del bounding_features_test, bounding_features_train
# gc.collect()

print('Generating Metadata Feats')
train, test = metadata_features(train, test)
#personal testing
print(train.info(), train.get_dtype_counts(), train.select_dtypes('object').columns.tolist())

print('Generating Breed Map', train.shape)
X_temp = breed_maps(train, test, labels_breed)

print('Pre trained IMGS FE')
denseNet121 = build_model_img()
pretrained_feats = image_feature(denseNet121, train, test)
#train test merged now...
del denseNet121
gc.collect()

X_temp = X_temp.merge(pretrained_feats, how='left', on='PetID')
print('nlp feats components thing..')

X_feat = nlp_features(X_temp)
print(train.info(), train.get_dtype_counts(), train.select_dtypes('object').columns.tolist())

print('Modelling Starts Now')

X_feat.drop(['metadata_metadata_panAngle_MEAN','metadata_metadata_tiltAngle_MEAN',
    'metadata_metadata_detectionConfidence_MEAN','metadata_metadata_landmarkingConfidence_MEAN',
    'metadata_metadata_rollAngle_MEAN'], axis=1, inplace=True)

#ran till here for now
########################################################################################
#call to build poor models haha

lgb_submission, lgb_oof_train, lgb_oof_test = run_lgbm(X_feat, test)
lgb_submission.to_csv("submission_lgb.csv", index=None)

imps = pd.read_csv('imps.csv')
drop_cols = imps[imps['importance_split'] == 0]['feature'].values.tolist()

params_xgb = {
    'eval_metric': 'rmse',
    'seed': 1337,
    'eta': 0.0123, #0.0123,
    'subsample': 0.7, #0.7
    'colsample_bytree': 0.6,#.75
    'silent': 1,
    'gamma' : 8,
    'max_depth' : 6,
    'tree_method': 'gpu_hist'
    
}
xgb_submission, xgb_oof_train, xgb_oof_test = run_xgb(X_feat.drop(drop_cols, axis=1), test, params_xgb) #saves the coeff files npy format
xgb_submission.to_csv("submission_xgb.csv", index=None)

#################################GAME OVER##############################################
# # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX Blend XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
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

submission = lgb_submission[['PetID']]
# coefficients_ = [1.645, 2.06829671,2.47963736,2.89427377] 
coefficients_ = np.load('coeff_.npy')
test_predictions = predict((lgb_oof_test.mean(axis=1) + xgb_oof_test.mean(axis=1))/2, coefficients_)
submission = pd.DataFrame({'PetID': test['PetID'].values, 'AdoptionSpeed': test_predictions})
submission['AdoptionSpeed'] = submission.AdoptionSpeed.astype(int)
submission.to_csv('submission.csv', index=False)

X_train = X_feat.loc[np.isfinite(X_temp.AdoptionSpeed), :]
X_test  = X_feat.loc[~np.isfinite(X_temp.AdoptionSpeed), :]

del X_temp
gc.collect()

print('train', X_train.shape, 'test', X_test.shape)

#disable this in final run to save time
X_train.to_csv('train_proc.csv', index=None)
X_test.to_csv('test_proc.csv', index=None)

# ##### NN SIDE ATTEMPT
import time
import gensim
from nltk.stem import PorterStemmer
ps = PorterStemmer()
from nltk.stem.lancaster import LancasterStemmer
lc = LancasterStemmer()
from nltk.stem import SnowballStemmer
sb = SnowballStemmer("english")

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, CuDNNLSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D
from keras.layers import Bidirectional, GlobalMaxPool1D, GlobalMaxPooling1D, GlobalAveragePooling1D, BatchNormalization
from keras.layers import Input, Embedding, Dense, Conv2D, MaxPool2D, concatenate, Convolution1D
from keras.layers import Reshape, Flatten, Concatenate, Dropout, SpatialDropout1D, GaussianDropout
from keras.optimizers import Adam
from keras.models import Model
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.callbacks import EarlyStopping

import sys
from os.path import dirname
from keras import initializers
from keras.engine import InputSpec, Layer
from keras import backend as K

import spacy

import tensorflow as tf
'''
Observations:
 – Focal loss did not really outperform standard CE loss on both balanced/imbalanced data. This is conceivable since the focal loss is designed for detection.
 – If we carefully tuned alpha and gamma, focal loss somehow handle imbalanced data well (despite the oscillating valid. acc.)
'''
#https://medium.com/@ManishChablani/focal-loss-for-dense-object-detection-paper-summary-79a030798e42
def focal_loss(gamma=2., alpha=4.):

    gamma = float(gamma)
    alpha = float(alpha)

    def focal_loss_fixed(y_true, y_pred):
        """Focal loss for multi-classification
        FL(p_t)=-alpha(1-p_t)^{gamma}ln(p_t)
        Notice: y_pred is probability after softmax
        gradient is d(Fl)/d(p_t) not d(Fl)/d(x) as described in paper
        d(Fl)/d(p_t) * [p_t(1-p_t)] = d(Fl)/d(x)
        Focal Loss for Dense Object Detection
        https://arxiv.org/abs/1708.02002

        Arguments:
            y_true {tensor} -- ground truth labels, shape of [batch_size, num_cls]
            y_pred {tensor} -- model's output, shape of [batch_size, num_cls]

        Keyword Arguments:
            gamma {float} -- (default: {2.0})
            alpha {float} -- (default: {4.0})

        Returns:
            [tensor] -- loss.
        """
        epsilon = 1.e-9
        y_true = tf.convert_to_tensor(y_true, tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, tf.float32)

        model_out = tf.add(y_pred, epsilon)
        ce = tf.multiply(y_true, -tf.log(model_out))
        weight = tf.multiply(y_true, tf.pow(tf.subtract(1., model_out), gamma))
        fl = tf.multiply(alpha, tf.multiply(weight, ce))
        reduced_fl = tf.reduce_max(fl, axis=1)
        
        return tf.reduce_mean(reduced_fl)
    
    return focal_loss_fixed

def load_glove(word_dict, lemma_dict):
    
    EMBEDDING_FILE = '../input/glove840b300dtxt/glove.840B.300d.txt'
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))
    embed_size = 300
    nb_words = len(word_dict)+1
    embedding_matrix = np.zeros((nb_words, embed_size), dtype=np.float32)
    unknown_vector = np.zeros((embed_size,), dtype=np.float32) - 1.
    print(unknown_vector[:5])
    for key in tqdm(word_dict):
        word = key
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[word_dict[key]] = embedding_vector
            continue
        word = key.lower()
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[word_dict[key]] = embedding_vector
            continue
        word = key.upper()
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[word_dict[key]] = embedding_vector
            continue
        word = key.capitalize()
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[word_dict[key]] = embedding_vector
            continue
        word = ps.stem(key)
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[word_dict[key]] = embedding_vector
            continue
        word = lc.stem(key)
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[word_dict[key]] = embedding_vector
            continue
        word = sb.stem(key)
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[word_dict[key]] = embedding_vector
            continue
        word = lemma_dict[key]
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[word_dict[key]] = embedding_vector
            continue
        embedding_matrix[word_dict[key]] = unknown_vector                    
    return embedding_matrix, nb_words

y_train = X_train['AdoptionSpeed']
X_train.drop(['AdoptionSpeed', 'RescuerID'], axis=1, inplace=True)
X_test.drop(['RescuerID'], axis=1, inplace=True)

cat_cols = ['Type', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2', 'Color3', 'MaturitySize', 
            'FurLength', 'Vaccinated', 'Dewormed', 'Sterilized', 'Health','Quantity', 'State']

other_cols = list(set(X_train.columns.tolist()) - set(cat_cols))
other_cols = other_cols
removed_col = []

for i in other_cols:
    if i in drop_cols and i not in cat_cols:
        try:
            other_cols.remove(i)
            removed_col.append(i)
            print(f'Dropped the col named {i}')
        except:
            print(f'Couldnt Remove the col {i} as it was present in something probably')
del removed_col
gc.collect()

print('Shape before,', X_train.shape, X_test.shape)

X_train = X_train[other_cols+cat_cols]
X_test  = X_test[other_cols+cat_cols]

print('Shape After,', X_train.shape, X_test.shape)

pic_cols, svd_cols, nmf_cols = [], [], []

for col in other_cols:
    if 'pic_' in col: 
        pic_cols.append(col)
        other_cols.remove(col)
    elif 'SVD_' in col:
        svd_cols.append(col)
        other_cols.remove(col)
    elif 'NMF_' in col:
        nmf_cols.append(col)
        other_cols.remove(col)        

col_vals_dict = {c: list(X_train[c].unique()) for c in cat_cols}
nb_numeric   = len(other_cols)
nb_categoric = len(col_vals_dict)
print('Number of Numerical features:', nb_numeric)
print('Number of Categorical features:', nb_categoric)

embed_cols = []
len_embed_cols = []
for c in col_vals_dict:
    embed_cols.append(c)
    len_embed_cols.append((c, len(col_vals_dict[c])))
    #print(c + ': %d values' % len(col_vals_dict[c])) #look at value counts to know the embedding dimensions
        
print('\n Number of embed features :', len(embed_cols)) #Type is Dropped
len(other_cols), len(embed_cols), X_train.shape

def preproc(X_train, X_test):

    input_list_train = []
    input_list_test   = []
    
    global other_cols, pic_cols
    
    
    #the cols to be embedded: rescaling to range [0, # values)
    for c in embed_cols:
        raw_tests = np.unique(X_train[c])
        test_map = {}
        for i in range(len(raw_tests)):
            test_map[raw_tests[i]] = i       
        input_list_train.append(X_train[c].map(test_map).values)
        input_list_test.append(X_test[c].map(test_map).fillna(0).values)
        
    #the rest of the columns
    input_list_train.append(X_train[other_cols].values)
    input_list_test.append(X_test[other_cols].values)
    
    #pic cols
    input_list_train.append(X_train[pic_cols].values)
    input_list_test.append(X_test[pic_cols].values)
    
    #nmf cols
    input_list_train.append(X_train[svd_cols].values)
    input_list_test.append(X_test[svd_cols].values)
    
    #nmf cols
    input_list_train.append(X_train[nmf_cols].values)
    input_list_test.append(X_test[nmf_cols].values)
    
    return input_list_train ,input_list_test

print('Len Embed Cols', len(embed_cols))

from sklearn.preprocessing import MinMaxScaler, StandardScaler

scaler  = MinMaxScaler()
scaler.fit(X_train[other_cols])

x_train_num_scale = scaler.transform(X_train[other_cols]) #lost col names here
x_test_num_scale  = scaler.transform(X_test[other_cols])

scaler  = MinMaxScaler()
scaler.fit(X_train[pic_cols])
x_train_pic_scale = scaler.transform(X_train[pic_cols]) #lost col names here
x_test_pic_scale  = scaler.transform(X_test[pic_cols])

scaler  = MinMaxScaler()
scaler.fit(X_train[svd_cols])
x_train_svd_scale = scaler.transform(X_train[svd_cols]) #lost col names here
x_test_svd_scale  = scaler.transform(X_test[svd_cols])

scaler  = MinMaxScaler()
scaler.fit(X_train[nmf_cols])
x_train_nmf_scale = scaler.transform(X_train[nmf_cols]) #lost col names here
x_test_nmf_scale  = scaler.transform(X_test[nmf_cols])

x_train_inp, x_test_inp = preproc(X_train, X_test)
del scaler
gc.collect()

nb_classes = 5

from keras.optimizers import *
import keras.backend as K
from keras.callbacks import Callback
from keras.optimizers import Optimizer
from keras import backend as K, initializers, regularizers, constraints
from keras.engine.topology import Layer

from keras.layers import K, Activation
from keras.engine import Layer, InputSpec
import tensorflow as tf

from keras.layers import add, dot
from itertools import combinations

from keras import backend as K, initializers, regularizers, constraints
from keras.engine.topology import Layer

class AdaBound(Optimizer):
    """AdaBound optimizer.
    Default parameters follow those provided in the original paper.
    # Arguments
        lr: float >= 0. Learning rate.
        final_lr: float >= 0. Final learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        gamma: float >= 0. Convergence speed of the bound function.
        epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
        decay: float >= 0. Learning rate decay over each update.
        amsbound: boolean. Whether to apply the AMSBound variant of this
            algorithm.
    # References
        - [Adaptive Gradient Methods with Dynamic Bound of Learning Rate]
          (https://openreview.net/forum?id=Bkg3g2R9FX)
        - [Adam - A Method for Stochastic Optimization]
          (https://arxiv.org/abs/1412.6980v8)
        - [On the Convergence of Adam and Beyond]
          (https://openreview.net/forum?id=ryQu7f-RZ)
    """

    def __init__(self, lr=0.001, final_lr=0.1, beta_1=0.9, beta_2=0.999, gamma=1e-3,
                 epsilon=None, decay=0., amsbound=False, weight_decay=0.0, **kwargs):
        super(AdaBound, self).__init__(**kwargs)

        if not 0. <= gamma <= 1.:
            raise ValueError("Invalid `gamma` parameter. Must lie in [0, 1] range.")

        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(decay, name='decay')

        self.final_lr = final_lr
        self.gamma = gamma

        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_decay = decay
        self.amsbound = amsbound

        self.weight_decay = float(weight_decay)
        self.base_lr = float(lr)

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))

        t = K.cast(self.iterations, K.floatx()) + 1

        # Applies bounds on actual learning rate
        step_size = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
                          (1. - K.pow(self.beta_1, t)))

        final_lr = self.final_lr * lr / self.base_lr
        lower_bound = final_lr * (1. - 1. / (self.gamma * t + 1.))
        upper_bound = final_lr * (1. + 1. / (self.gamma * t))

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        if self.amsbound:
            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        else:
            vhats = [K.zeros(1) for _ in params]
        self.weights = [self.iterations] + ms + vs + vhats

        for p, g, m, v, vhat in zip(params, grads, ms, vs, vhats):
            # apply weight decay
            if self.weight_decay != 0.:
                g += self.weight_decay * K.stop_gradient(p)

            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)

            if self.amsbound:
                vhat_t = K.maximum(vhat, v_t)
                denom = (K.sqrt(vhat_t) + self.epsilon)
                self.updates.append(K.update(vhat, vhat_t))
            else:
                denom = (K.sqrt(v_t) + self.epsilon)

            # Compute the bounds
            step_size_p = step_size * K.ones_like(denom)
            step_size_p_bound = step_size_p / denom
            # TODO: Replace with K.clip after releast of Keras > 2.2.4
            bounded_lr_t = m_t * tf.clip_by_value(step_size_p_bound,
                                                  lower_bound,
                                                  upper_bound)

            p_t = p - bounded_lr_t

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'final_lr': float(self.final_lr),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'gamma': float(self.gamma),
                  'decay': float(K.get_value(self.decay)),
                  'epsilon': self.epsilon,
                  'weight_decay': self.weight_decay,
                  'amsbound': self.amsbound}
        base_config = super(AdaBound, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def build_model(embedding_matrix, nb_words, loss = None, embedding_size=300):
    
    global other_cols, cat_cols, len_embed_cols
    
    model_out = []
    model_in  = []
    filter_sizes = [1,2,3,5]
    num_filters = 32
    
    inp = Input(shape=(max_length,)) #1st
    x = Embedding(nb_words, embedding_size, weights=[embedding_matrix], trainable=False)(inp)
    x = SpatialDropout1D(0.3)(x)
    # res = Reshape((max_length, embedding_size, 1))(x) with cnn2d
    #bidir model
    x1 = Bidirectional(CuDNNLSTM(256, return_sequences=True))(x)#should this be changed ????
    x2 = Bidirectional(CuDNNGRU(128, return_sequences=True))(x)##should this be changed ????
    max_pool1 = GlobalMaxPooling1D()(x1)##should this be changed ????
    max_pool2 = GlobalMaxPooling1D()(x2)##should this be changed ????
    conc = Concatenate()([max_pool1, max_pool2])##should this be changed ???? Not sure (NLP idea)
    # cnn model approach
    
    # conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embedding_size), kernel_initializer='normal', activation='relu')(res)
    # conv_3 = Conv2D(num_filters, kernel_size=(filter_sizes[3], embedding_size), kernel_initializer='normal', activation='relu')(res)
    # maxpool_1 = MaxPool2D(pool_size=(max_length - filter_sizes[1] + 1, 1))(conv_1)
    # maxpool_3 = MaxPool2D(pool_size=(max_length - filter_sizes[3] + 1, 1))(conv_3)
    # z = Concatenate(axis=1)([maxpool_1, maxpool_3])   
    # z = Flatten()(z)
    # z = Dropout(0.1)(z)
    
    model_in.append(inp)
    
    #2nd inp cat_cols
    
    for name, dim in len_embed_cols:
        
        input_dim = Input(shape=(1,), dtype='int32')
        embed_dim = Embedding(dim, min(50, dim//2 + 1), input_length= 1, name = name, trainable=True)(input_dim) #1 for unknowns
        embed_dim = Dropout(0.2)(embed_dim)
        embed_dim = Flatten()(embed_dim) #Tilli's Help/His share https://www.kaggle.com/c/microsoft-malware-prediction/discussion/76384
        model_out.append(embed_dim)
        model_in.append(input_dim)
    
    num = Input(shape=(np.asarray(other_cols).shape[0],), dtype= 'float32')
    x = Dense(256, activation='relu')(num)
    x = Dropout(.2)(x)
    x = BatchNormalization()(x)
    
    pics = Input(shape=(np.array(pic_cols).shape[0],), dtype='float32')
    x_pic = Dense(256, activation='relu')(pics)
    x_pic = Dropout(.2)(x_pic)
    x_pic = BatchNormalization()(x_pic)
    
    svd = Input(shape=(np.array(svd_cols).shape[0],), dtype='float32')
    x_svd = Dense(256, activation='relu')(svd)
    x_svd = Dropout(.25)(x_svd)
    x_svd = BatchNormalization()(x_svd)
    
    nmf = Input(shape=(np.array(nmf_cols).shape[0],), dtype='float32')
    x_nmf = Dense(256, activation='relu')(nmf)
    x_nmf = Dropout(.25)(x_nmf)
    x_nmf = BatchNormalization()(x_nmf)
    
    txt_desc = Concatenate(axis=1)([x_svd, x_nmf])
    outputs = Concatenate(axis=1)([conc, *model_out, x, x_pic, txt_desc])
    
    x = Dense(256,activation='relu', kernel_regularizer=regularizers.l2(0.0001))(outputs)
    x = Dropout(.25)(x)
    x = BatchNormalization()(x)
    x = Dense(16,activation='relu')(x)
    x = Dropout(.1)(x)
    predictions = Dense(nb_classes, activation='softmax', name='outputs')(x)
    
    model = Model(inputs=[*model_in, num, pics, svd, nmf], outputs=predictions,)
    from keras import losses
    optm  = AdaBound(lr=1e-03, final_lr=0.1, gamma=1e-03, weight_decay=1e-4, amsbound=False)
    # adam = optimizers.SGD(lr=1e-03, momentum=0.9)
    model.compile(optimizer=optm, loss=loss, metrics=['accuracy'])
    
    #https://blog.zenggyu.com/en/post/2018-12-05/retinanet-explained-and-demystified/#fnref10
    ###Need to play around the focal loss params maybe
    
    return model

start_time = time.time()
print("Loading data ...")

train_text = train['Description'].astype(str)
test_text = test['Description'].astype(str)

text_list = pd.concat([train_text, test_text])
y = y_train
num_train_data = y.shape[0]
print("--- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
print("Spacy NLP ...")
nlp = spacy.load('en_core_web_lg', disable=['parser','ner','tagger'])
nlp.vocab.add_flag(lambda s: s.lower() in spacy.lang.en.stop_words.STOP_WORDS, spacy.attrs.IS_STOP)
word_dict = {}
word_index = 1
lemma_dict = {}
docs = nlp.pipe(text_list, n_threads = 4)
word_sequences = []
for doc in docs:
    word_seq = []
    for token in doc:
        if (token.text not in word_dict) and (token.pos_ is not "PUNCT"):
            word_dict[token.text] = word_index
            word_index += 1
            lemma_dict[token.text] = token.lemma_
        if token.pos_ is not "PUNCT":
            word_seq.append(word_dict[token.text])
    word_sequences.append(word_seq)

del docs
gc.collect()

train_word_sequences = word_sequences[:num_train_data]
test_word_sequences = word_sequences[num_train_data:]

print("--- %s seconds ---" % (time.time() - start_time))

# hyperparameters
max_length = 55 #this needs to be determined via stats of texts and how much embeddings cover for us
embedding_size = 300 #no stacking as of now but its the same way and Quora kernel
learning_rate = 0.001
batch_size = 64
num_epoch = 100 #haha let'see what happens

train_word_sequences = pad_sequences(train_word_sequences, maxlen=max_length, padding='post')
test_word_sequences  = pad_sequences(test_word_sequences,  maxlen=max_length, padding='post')
print(train_word_sequences[:1])
print(test_word_sequences[:1])

pred_prob = np.zeros((len(test_word_sequences),), dtype=np.float32)

del X_train, X_test
gc.collect()

start_time = time.time()
print("Loading embedding matrix ...")
embedding_matrix, nb_words = load_glove(word_dict, lemma_dict)
print("--- %s seconds ---" % (time.time() - start_time))

from keras.utils import to_categorical
gc.collect()
K.clear_session()
start_time = time.time()
print("Start training ...")

model = build_model(embedding_matrix, nb_words, focal_loss(2.75, 0.85), embedding_size)
es = EarlyStopping(monitor='val_loss', mode='min',patience= 30, verbose=1, restore_best_weights=True)

print('Training...')

history = model.fit([train_word_sequences, np.array(x_train_inp[0]),np.array(x_train_inp[1]),np.array(x_train_inp[2]),np.array(x_train_inp[3]),np.array(x_train_inp[4]),np.array(x_train_inp[5]),\
          np.array(x_train_inp[6]),np.array(x_train_inp[7]),np.array(x_train_inp[8]),np.array(x_train_inp[9]),np.array(x_train_inp[10]),\
          np.array(x_train_inp[11]),np.array(x_train_inp[12]),np.array(x_train_inp[13]),np.array(x_train_inp[14]), \
          x_train_num_scale, x_train_pic_scale, x_train_svd_scale, x_train_nmf_scale], \
          to_categorical(y.values, num_classes=5), batch_size=batch_size, epochs= num_epoch, verbose=2, validation_split=0.1,\
          callbacks=[es])

del train_word_sequences, es, text_list, test_text, train_text, x_train_num_scale, x_train_pic_scale, x_train_svd_scale, x_train_nmf_scale
gc.collect()

pred_prob = model.predict([test_word_sequences, np.array(x_test_inp[0]),np.array(x_test_inp[1]),np.array(x_test_inp[2]),np.array(x_test_inp[3]),np.array(x_test_inp[4]),np.array(x_test_inp[5]),\
          np.array(x_test_inp[6]),np.array(x_test_inp[7]),np.array(x_test_inp[8]),np.array(x_test_inp[9]),np.array(x_test_inp[10]),\
          np.array(x_test_inp[11]),np.array(x_test_inp[12]),np.array(x_test_inp[13]),np.array(x_test_inp[14]),\
          x_test_num_scale, x_test_pic_scale, x_test_svd_scale, x_test_nmf_scale],\
          batch_size=batch_size, verbose=2)

pred_probs = np.clip(pred_prob, -0.99,4.99)
del pred_prob
gc.collect()

print("--- %s seconds ---" % (time.time() - start_time))
print('Min, max', pred_probs.min(), pred_probs.max())
np.save('pred_prob_nn.npy', pred_probs)

del history, embedding_matrix, word_sequences, nb_words, model,test_word_sequences
gc.collect()

print(np.argmax(pred_probs, axis=1))
preds = np.argmax(pred_probs, axis=1)
np.save('preds_NN.npy', preds)
print(preds)

submission = lgb_submission[['PetID']]
# coefficients_ = [1.645, 2.06829671,2.47963736,2.89427377] 
coefficients_ = np.load('coeff_.npy')
test_predictions = predict((lgb_oof_test.mean(axis=1) + xgb_oof_test.mean(axis=1) + preds)/3., coefficients_)
submission = pd.DataFrame({'PetID': test['PetID'].values, 'AdoptionSpeed': test_predictions})
submission['AdoptionSpeed'] = submission.AdoptionSpeed.astype(int)
submission.to_csv('submission_lgb_xgb_nn_xgbsoft.csv', index=False)

print('Done.....!!')
print(submission)

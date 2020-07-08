#!/usr/bin/env python
# coding: utf-8

import os

path = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(path, "data")
model_dir = os.path.join(path, "model")

CRAWL_EMBEDDING_PATH = os.path.join(data_dir, 'crawl-300d-2M.pkl') # i expect the embeddings to be in data folder
GLOVE_EMBEDDING_PATH = os.path.join(data_dir, 'glove.840B.300d.pkl') # i expect the embeddings to be in data folder
word_embeddings_url1 = # Refer_Kaggle_Datasets https://www.kaggle.com/authman/pickled-crawl300d2m-for-kernel-competitions
word_embeddings_url2 = # Refer_Kaggle_Datasets https://www.kaggle.com/authman/pickled-glove840b300d-for-10sec-loading

# arch 1 LSTM Based
LSTM_UNITS = 256
DENSE_HIDDEN_UNITS = 4 * LSTM_UNITS
MAX_LEN = 300

# arch 2 CNN Based
N_FILTERS = 256
FILTER_SIZES = [2,3,4,5,6]

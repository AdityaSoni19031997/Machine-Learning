#!/usr/bin/env python
# coding: utf-8

import torch
import gc
import pandas as pd

from keras.preprocessing import text, sequence
from torch import nn
from torch.utils import data
from torch.nn import functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

from tcs_scripts_sol import data_dir, model_dir, CRAWL_EMBEDDING_PATH, GLOVE_EMBEDDING_PATH, 
from tcs_scripts_sol import LSTM_UNITS, DENSE_HIDDEN_UNITS ,MAX_LEN ,N_FILTERS, FILTER_SIZES
from tcs_scripts_sol.preprocess import meta_nlp_feats, handle_punctuation, convert_emoticons, chat_words_conversion, handle_contractions, fix_quote
from tcs_scripts_sol.model import NeuralNet, CNN, TabularDataset, FocalLoss
from tcs_scripts_sol.utils import seed_everything, build_matrix, fit

class Train:
    def __init__(self, batch_size=256, epochs=None, learning_rate=None, fc_dropout=None, model_name=None, output_size=None):

        self.learning_rate = learning_rate
        self.fc_dropout = fc_dropout
        self.batch_size = batch_size
        self.epochs = epochs
        self.model_name = model_name
        self.output_size = output_size

    def run_pipeline(self):
    	
        # start pipeline
        data = load_data(fname='sentiment.csv')
        data = process_data(data=data)
        data = split_data(data=data)
        _ = construct_model(data=data)

    def load_data(self, fname='sentiment.csv'):

    	f_path = os.path.join(data_dir, fname)
        df  = pd.read_csv(f_path)
        return df

    def process_data(self, data):
    	
    	df = data
    	scaler = StandardScaler()
    	df = meta_nlp_feats(df, 'content')
    	df['content'] = df['content'].apply(lambda x:chat_words_conversion(x))
    	df['content'] = df['content'].apply(lambda x:convert_emoticons(x))
    	df['content'] = df['content'].apply(lambda x:handle_punctuation(x))
    	df['content'] = df['content'].apply(lambda x:handle_contractions(x))
    	df['content'] = df['content'].apply(lambda x:fix_quote(x.split()))
        
        x_train = df['content']
        y_aux_train = df['label'].values
        meta_feats_lst = scaler.fit_transform(df.values[:,5:])
        
        tokenizer = text.Tokenizer(num_words = 100000, filters='', lower=False)
        tokenizer.fit_on_texts(list(x_train))
        x_train = tokenizer.texts_to_sequences(x_train)
        with open('./token_stats.json', 'w') as outfile:
        	print('Saving tokenizer....')
        	json.dump(tokenizer.get_config(), outfile)

        lengths = np.array([len(x) for x in x_train])
        x_train = sequence.pad_sequences(x_train, maxlen=max(lengths))

        crawl_matrix, unknown_words_crawl = build_matrix(tokenizer.word_index, CRAWL_EMBEDDING_PATH)
        glove_matrix, unknown_words_glove = build_matrix(tokenizer.word_index, GLOVE_EMBEDDING_PATH)
        
        max_features = len(tokenizer.word_index) + 1
        embedding_matrix = np.concatenate([crawl_matrix, glove_matrix], axis=-1)
        
        del crawl_matrix, glove_matrix, lengths
        gc.collect()

        return x_train, y_aux_train, meta_feats_lst, max_features, embedding_matrix

    def split_data(self, data):
    	
    	x_train, y_aux_train, meta_feats_lst, max_features, embedding_matrix = data
    	all_idx = np.random.permutation(30000)
    	train_idx, val_idx = all_idx[:25000], all_idx[25000:]
    	
    	dataset_train = TabularDataset(x_train[train_idx], y_aux_train[train_idx], meta_feats_lst[train_idx])
    	dataset_val = TabularDataset(x_train[val_idx], y_aux_train[val_idx], meta_feats_lst[val_idx])
    	train_dl = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
    	val_dl = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False)

        return train_dl, val_dl, data

    def construct_model(self, data):

        train_dl, val_dl, config = data
        x_train, y_aux_train, meta_feats_lst, max_features, embedding_matrix = config

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model_name =='NeuralNet':
        	model = NeuralNet(embedding_matrix, num_aux_targets=self.output_size)
        else:
        	model = CNN(embedding_matrix, max_features=max_features, n_filters=N_FILTERS, filter_sizes=FILTER_SIZES, output_dim=self.output_size)
        seed_everything(seed=1234)

        criterion = FocalLoss(gamma=3, eps=1e-5)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        model.to(device)

        print(f'Trained model will be cached to {model_dir}')
        model_path = os.path.join(model_dir, "model.pkl")

        lr, tloss, vloss = fit(
        	model=model, train_dl=train_dl, val_dl=val_dl, loss_fn=criterion, 
        	opt=optimizer, scheduler=None, device=device, epochs=self.epochs, 
        	model_path=model_path,
        	)
        return None

def main( epochs, batch_size, learning_rate, fc_dropout, model_name, output_size):
    pipeline = Train(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, fc_dropout=fc_dropout, model_name='CNN', output_size=5)
    pipeline.run_pipeline()

if __name__ == "__main__":
    main()

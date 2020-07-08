#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import torch
import fastai
import time
import gc
import random
import operator 

from keras.preprocessing import text, sequence
from tqdm import tqdm

def seed_everything(seed=1234):

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) #should actually be done before
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')

def load_embeddings(path):

    with open(path,'rb') as f:
        emb_arr = pickle.load(f)
    return emb_arr

def build_matrix(word_index, path):

    embedding_index = load_embeddings(path)
    embedding_matrix = np.zeros((len(word_index) + 1, 300))
    unknown_words = []
    
    for word, i in word_index.items():
        try:
            embedding_matrix[i] = embedding_index[word]
        except KeyError:
            unknown_words.append(word)
    return embedding_matrix, unknown_words

def check_coverage(vocab,embeddings_index):

    a, oov, k, i = {} , {}, 0, 0
    for word in tqdm(vocab):
        try:
            a[word] = embeddings_index[word]
            k += vocab[word]
        except:

            oov[word] = vocab[word]
            i += vocab[word]
            pass

    print('Found embeddings for {:.2%} of vocab'.format(len(a) / len(vocab)))
    print('Found embeddings for  {:.2%} of all text'.format(k / (k + i)))
    sorted_x = sorted(oov.items(), key=operator.itemgetter(1))[::-1]

    return sorted_x

def build_vocab(sentences, verbose = True):

    vocab = {}
    for sentence in tqdm(sentences, disable = (not verbose)):
        for word in sentence:
            try:
                vocab[word] += 1
            except KeyError:
                vocab[word] = 1
    return vocab

def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)

def fit(model, train_dl, val_dl, loss_fn, opt, scheduler=None, device='cuda', epochs=10, model_path=None, best_val_loss=None, verbose=True):
    
    print(f'The model has {count_parameters(model):,} trainable parameters')
    
    lr = defaultdict(list)
    tloss = defaultdict(list)
    vloss = defaultdict(list)
    
    num_batch = len(train_dl)
    train_metrics = ClassificationMeter()
    val_metrics = ClassificationMeter()
    
    for epoch in range(epochs):

        y_true_train = list()
        y_pred_train = list()
        total_loss_train = 0
        model.train();

        t = tqdm(iter(train_dl), leave=False, total=num_batch, desc=f'Epoch {epoch}/{epochs-1}')
        
        for y, text_seq, cat in t:
            if scheduler is not None:
                scheduler.batch_step()
            cat = cat.to(device, dtype=torch.float)
            text_seq = text_seq.to(device)
            y = y.to(device)
            t.set_description(f'Epoch {epoch}/{epochs-1}')
            opt.zero_grad()
            pred = model(text_seq, cat)
            loss = loss_fn(pred, y)
            loss.backward()
            lr[epoch].append(opt.param_groups[0]['lr'])
            tloss[epoch].append(loss.item())
            opt.step()
            t.set_postfix(loss=np.mean(tloss[epoch]))

            y_true_train += list(y.cpu().data.numpy())
            y_pred_train += list(np.argmax(pred.cpu().data.numpy(), axis=1))
            total_loss_train += loss.item()

        train_loss = total_loss_train / len(train_dl)
        train_metrics.update(y_pred_train, y_true_train)
        train_acc = train_metrics.acc
    
        if val_dl:
            
            model.eval()
            y_true_val = list()
            y_pred_val = list()
            total_loss_val = 0
            
            for y, text_seq, cat in tqdm(val_dl, leave=False):    
                cat = cat.to(device, dtype=torch.float)
                text_seq = text_seq.to(device)
                y = y.to(device)
                pred = model(text_seq, cat)
                text_seq = None
                cat = None
                loss = loss_fn(pred, y)
                y_true_val += list(y.cpu().data.numpy())
                y_pred_val += list(np.argmax(pred.cpu().data.numpy(), axis=1))
                total_loss_val += loss.item()
                vloss[epoch].append(loss.item())

            valloss = total_loss_val / len(val_dl)
            val_metrics.update(y_pred_val, y_true_val)
            valacc = val_metrics.acc
        
        if verbose:
            print(f'Epoch {epoch}/{epochs-1}: train_loss: {train_loss:.4f} '
                  f'train_acc: {train_acc:.4f} | val_loss: {valloss:.4f} val_acc: {valacc:.4f}'
                 )
    return lr, tloss, vloss

class ClassificationMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.acc = 0
        self.pre = 0
        self.rec = 0
        self.f1 = 0
        self.bacc = 0

    def update(self, output, target):
        
        self.acc = metrics.accuracy_score(target, output)
        self.bacc = metrics.balanced_accuracy_score(target, output)
        self.pre = metrics.precision_score(target, output, average='micro') # micro is preferred with imbalanced
        self.rec = metrics.recall_score(target, output, average='micro') # micro is preferred with imbalanced
        self.f1 = metrics.f1_score(target, output, average='micro') # micro is preferred with imbalanced

def load_previous_tokenizer(cache_dir):
    
    print('loading tokenizer')
    with open(f'./{cache_dir}/token_stats.json') as f:
        previous_tokenizer = json.load(f)

    tokenizer = text.Tokenizer()
    tokenizer.num_words  = previous_tokenizer['num_words']
    tokenizer.filters    = previous_tokenizer['filters']
    tokenizer.lower      = previous_tokenizer['lower']
    tokenizer.split      = previous_tokenizer['split']
    tokenizer.char_level = previous_tokenizer['char_level']
    tokenizer.oov_token  = previous_tokenizer['oov_token']
    tokenizer.word_docs  = previous_tokenizer['word_docs']
    tokenizer.index_docs = previous_tokenizer['index_docs']
    tokenizer.index_word = previous_tokenizer['index_word']
    tokenizer.document_count = previous_tokenizer['document_count']
    tokenizer.word_counts    = previous_tokenizer['word_counts']
    tokenizer.word_index = json.loads(previous_tokenizer['word_index'])
    
    return tokenizer

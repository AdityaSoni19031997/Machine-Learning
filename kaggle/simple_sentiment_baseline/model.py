#!/usr/bin/env python
# coding: utf-8

import torch

from torch import nn
from torch.utils.data import Dataset
from torch.nn import functional as F

class TabularDataset(Dataset):
    def __init__(self, x_train_padded, y, meta_nlp):

        self.n = len(x_train_padded)
        self.text_seq = x_train_padded
        self.meta_nlp = meta_nlp
        self.y = y

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return [self.y[idx], self.text_seq[idx], self.meta_nlp[idx]]

class SpatialDropout(nn.Dropout2d):
    def forward(self, x):
        x = x.unsqueeze(2)    # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)
        return x

class NeuralNet(nn.Module):
    def __init__(self, embedding_matrix, num_aux_targets):
        super(NeuralNet, self).__init__()
        embed_size = embedding_matrix.shape[1]
        
        self.embedding = nn.Embedding(max_features, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.embedding_dropout = SpatialDropout(0.3)
        self.dropout = nn.Dropout(0.6) #0.5 gave 55
        self.dropout_last = nn.Dropout(0.3)
        
        self.lstm1 = nn.LSTM(embed_size, LSTM_UNITS, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(LSTM_UNITS * 2, LSTM_UNITS, bidirectional=True, batch_first=True)
    
        self.linear1 = nn.Linear(DENSE_HIDDEN_UNITS, DENSE_HIDDEN_UNITS)
        self.linear2 = nn.Linear(DENSE_HIDDEN_UNITS, DENSE_HIDDEN_UNITS)
        self.linear3 = nn.Linear(DENSE_HIDDEN_UNITS, 64)
        self.linear4 = nn.Linear(64+12, 16)
        self.linear_out = nn.Linear(16, num_aux_targets)
        
    def forward(self, x, meta_nlp):

        h_embedding = self.embedding(x.long())
        h_embedding = self.embedding_dropout(h_embedding)
        
        h_lstm1, _ = self.lstm1(h_embedding)
        h_lstm2, _ = self.lstm2(h_lstm1)
        
        avg_pool = torch.mean(h_lstm2, 1)
        max_pool, _ = torch.max(h_lstm2, 1)
        
        h_conc = torch.cat((max_pool, avg_pool), 1)
        h_conc_linear1  = F.relu(self.dropout(self.linear1(h_conc)))
        h_conc_linear2  = F.relu(self.dropout(self.linear2(h_conc)))
        
        hidden = h_conc + h_conc_linear1 + h_conc_linear2
        temp = self.linear3(hidden)
        temp = torch.cat((temp, meta_nlp), 1)
        result = self.linear_out((self.dropout_last(self.linear4(temp))))
        #result = self.linear_out(F.relu(self.dropout(self.linear4(temp))))
        return result

class FocalLoss(nn.Module):

    def __init__(self, gamma=2., eps=1e-5):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, input, target):
        y = one_hot_embedding(target, input.size(-1))
        logit = F.softmax(input)
        logit = logit.clamp(self.eps, 1. - self.eps)
        y = y.to('cuda')
        logit = logit.to('cuda')
        
        loss = -1 * y * torch.log(logit) # cross entropy
        loss = loss * (1 - logit) ** self.gamma # focal loss
        return loss.sum(dim=1).mean()

class CNN(nn.Module):
    def __init__(self, embedding_matrix, max_features=max_features, n_filters=None, filter_sizes=None, output_dim=5):
        '''
        when using text we only have a single channel, the text itself. 
        The out_channels is the number of filters and the kernel_size is the size of the filters. 
        Each of our kernel_sizes is going to be [n x emb_dim] where n is the size of the n-grams.
        As our model has N_FILTERS filters of len(FILTER_SIZES) different sizes, 
        that means we have N_FILTERS different n-grams the model thinks are important. 
        We concatenate these together into a single vector and pass them through a linear layer to predict the sentiment. 
        We can think of the weights of this linear layer as "weighting up the evidence" from each of the N_FILTERS n-grams 
        and making a final decision.
        '''
        super().__init__()
        
        embed_size = embedding_matrix.shape[1]
        self.embedding = nn.Embedding(max_features, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.embedding_dropout = SpatialDropout(0.3)
        
        # more generic and take any number of filters.
        self.convs = nn.ModuleList([
                                    nn.Conv2d(in_channels = 1, 
                                              out_channels = n_filters, 
                                              kernel_size = (fs, embed_size)) 
                                    for fs in filter_sizes
                                    ])
        
        self.fc = nn.Linear(len(filter_sizes) * n_filters + 12, 64)
        self.fc_out = nn.Linear(64, output_dim)
        self.dropout = nn.Dropout(0.5) #was 0.4
        
    def forward(self, text, meta_nlp):
        
        embedded = self.embedding_dropout(self.embedding(text.long()))
        # embedded = [batch_size, sent_len, emb dim]
        embedded = embedded.unsqueeze(1)
        # embedded = [batch_size, 1, sent_len, emb dim]
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        # conved_n = [batch_size, n_filters, sent_len - filter_sizes[n] + 1]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        # pooled_n = [batch_size, n_filters]
        cat = F.relu(self.dropout(torch.cat((*pooled, meta_nlp), dim = 1)))
        # cat = [batch_size, n_filters * len(filter_sizes)]
        output = self.fc_out(self.dropout(self.fc(cat)))
        return output

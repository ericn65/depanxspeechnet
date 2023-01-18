'''
In this script I'll implement my approach to DepAudioNet.
The inputs to this network are labeled spectrograms. At this point I'm not considering applying extra-info as the age.
The network consists on:
    - ResCNN with num_layers as a Hyperparameter to optimize. Convolutions will be 1D. Batch normalization can be considered.
    - LSTM layer with attention unidirectional.
    - FC Layer to classify between the different classes. 
The output will be softmaxed. 
'''
import math
import numpy as np
import torch
import torch.nn as nn 
import torch.nn.functional as F 

class ResCNN(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample= None):
        super(ResCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=1)
        )
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        residual = x
        output = self.conv1(x)
        output = self.conv2(output)
        if self.downsample:
            residual = self.downsample(x)
        output += residual
        output = self.relu(output)
        #output = self.dropout(output)
        return output

class LSTM(nn.Module):
    def __init__(self, n_rnn_dim, h_rnn_layer, n_rnn_layers):
        super(LSTM,self).__init__()
        self.lstm = nn.LSTM(
            input_size= n_rnn_dim,
            hidden_size= h_rnn_layer,
            num_layers= n_rnn_layers,
            batch_first=True,
            bidirectional=True
        )

    def forward(self,x):
        x = self.lstm(x)
        return x

class Attention(nn.Module):
    def __init__(self, hidden_size, batch_first=False):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.batch_first = batch_first

        self.att_weights = nn.parameter.Parameter(torch.Tensor(1,hidden_size),requires_grad=True)
        stdv = 1.0/np.sqrt(self.hidden_size)
        for weight in self.att_weights:
            nn.init.uniform_(weight, -stdv, stdv)
        
    def get_mask(self):
        pass

    def forward(self, inputs, lengths):
        if self.batch_first:
            batch_size, max_len = inputs.size()[:2]
        else:
            max_len, batch_size = inputs.size()[:2]
        
        weights = torch.bmm(inputs,
                            self.att_weights
                            .permute(1,0)
                            .unsqueeze(0)
                            .repeat(batch_size, 1,1)
                            )
        attentions = torch.softmax(F.relu(weights.squeeze()))
        mask = torch.ones(attentions.size(), requires_grad=True)
        for i, l in enumerate(lengths):
            if l < max_len:
                mask[i,l:] = 0
        masked = attentions*mask
        _sums = masked.sum(-1).unsqueeze(-1)

        attentions = masked.div(_sums)
        weighted = torch.mul(inputs, attentions.unsqueeze(-1).expand_as(inputs))
        representations = weighted.sum(1).squeeze()

        return representations, attentions

class MyLSTM(nn.Module):
    def __init__(self, n_rnn_dim, h_rnn_layer, n_rnn_layers, dropout=0.2):
        super(MyLSTM, self).__init__()
        self.dropout = nn.Dropout(p=0.2)
        self.lstm1 = LSTM(
            n_rnn_dim=n_rnn_dim,
            h_rnn_layer=h_rnn_layer,
            n_rnn_layers=n_rnn_layers
        )
        self.atten1 = Attention(h_rnn_layer*2, batch_first=True)
        self.lstm2 = LSTM(
            n_rnn_dim=h_rnn_layer*2,
            h_rnn_layer=h_rnn_layer,
            n_rnn_layers=1
        )
        self.atten2 = Attention(h_rnn_layer*2, batch_first=True)
        
    def forward(self,x):
        x = self.dropout(x)
        out1 = self.lstm1(x)
        x, lengths = nn.utils.rnn.pad_packed_sequence(out1,batch_first=True)
        x = self.atten1(x, lengths)
        out2 = self.lstm2(out1)
        y, lengths = nn.utils.rnn.pad_packed_sequence(out2, batch_first=True)
        y = self.atten2(y, lengths)
        z = torch.cat([x,y], dim=1)
        return z

class FullyConnected(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FullyConnected, self).__init__()
        self.fc = nn.Linear(in_features=in_channels, out_features=out_channels)
        self.activation = nn.ReLU()
        self.bn = nn.BatchNorm1d(in_channels)
    
    def forward(self, x):
        #x = self.bn(x)
        x = self.fc(x)
        x = self.activation(x)
        return x

class AttentionModel(nn.Module):
    def __init__(self, n_cnn_layers, n_rnn_dim, h_rnn_layer, n_rnn_layers, n_classes, stride=2, dropout=0.1):
        super(AttentionModel, self).__init__()
        self.dense = nn.Conv1d(500, 32, 1)
        self.cnn = nn.Sequential(*[
            ResCNN(
                32,
                32
            ) for i in range(n_cnn_layers)
        ])
        self.dense2 = nn.Linear(
            32*40,
            n_rnn_dim
        )
        self.lstm = nn.LSTM(
            input_size= n_rnn_dim,
            hidden_size= h_rnn_layer,
            num_layers= n_rnn_layers,
            batch_first=True,
            bidirectional=True
        )
        self.atten = MyLSTM(
            n_rnn_dim= n_rnn_dim, 
            h_rnn_layer= h_rnn_layer,
            n_rnn_layers= n_rnn_layers
        )
        self.fc = FullyConnected(
            in_channels= n_rnn_dim,
            out_channels= n_classes
        )

    def forward(self, x):
        batch,freq,width = x.shape
        x = self.dense(x)
        x = self.cnn(x)
        x = x.view(x.size(0),x.size(1)*x.size(2))
        x = self.dense2(x)
        #x = x.transpose(0,1)
        x,_ = self.lstm(x)
        #x = self.atten(x)
        x = x.transpose(0,1)
        #x = x[:,-1]
        x = x.transpose(0,1)
        x = self.fc(x)
        return x


class AnxietyFromDepression(nn.Module):
    def __init__(self, pretrained_model):
        super(AnxietyFromDepression, self).__init__()
        self.pretrained = pretrained_model
        self.new_layer = nn.Linear(5,4)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        x = self.pretrained(x)
        x = self.activation(x)
        x = self.new_layer(x)
        return x

class Scripted2Unscripted(nn.Module):
    def __init__(self, pretrained_model):
        super(Scripted2Unscripted, self).__init__()
        self.pretrained = pretrained_model

    def forward(self,x):
        x = self.pretrained(x)
        return x

class HydraNetTL(nn.Module):
    def __init__(self,pretrained_model):
        super(HydraNetTL, self).__init__()
        self.pretrained = pretrained_model
        self.fcAnx = nn.Linear(5,4)
        self.activation = nn.ReLU()
    
    def forward(self,x):
        x = self.pretrained(x)
        phq8 = x
        gad7 = self.activation(x)
        gad7 = self.fcAnx(gad7)
        return phq8, gad7

class HydraNet(nn.Module):
    def __init__(self, n_cnn_layers, n_rnn_dim, h_rnn_layer, n_rnn_layers, n_classes, stride=2, dropout=0.1):
        super(HydraNet, self).__init__()
        self.dense = nn.Conv1d(500, 32, 1)
        self.cnn = nn.Sequential(*[
            ResCNN(
                32,
                32
            ) for i in range(n_cnn_layers)
        ])
        self.dense2 = nn.Linear(
            32*40,
            n_rnn_dim
        )
        self.lstm = nn.LSTM(
            input_size= n_rnn_dim,
            hidden_size= h_rnn_layer,
            num_layers= n_rnn_layers,
            batch_first=True,
            bidirectional=True
        )
        self.atten = MyLSTM(
            n_rnn_dim= n_rnn_dim, 
            h_rnn_layer= h_rnn_layer,
            n_rnn_layers= n_rnn_layers
        )
        self.fcDep = FullyConnected(
            in_channels= n_rnn_dim,
            out_channels= n_classes
        )
        self.fcAnx = FullyConnected(
            in_channels= n_rnn_dim,
            out_channels= n_classes - 1
        )

    def forward(self, x):
        batch,freq,width = x.shape
        x = self.dense(x)
        x = self.cnn(x)
        x = x.view(x.size(0),x.size(1)*x.size(2))
        x = self.dense2(x)
        #x = x.transpose(0,1)
        x,_ = self.lstm(x)
        #x = self.atten(x)
        x = x.transpose(0,1)
        #x = x[:,-1]
        x = x.transpose(0,1)
        phq8 = self.fcDep(x)
        gad7 = self.fcAnx(x)
        return phq8, gad7

'''
This script contains the different network I'm using.
It also defines the model.
Attention is not yet used, but it could improve the model.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F 

class CNNLayerNorm(nn.Module):
    #Layer normalization built for CNNs input
    def __init__(self, n_feats):
        super(CNNLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(n_feats)

    def forward(self, x):
        x = x.transpose(1,2).contiguous()
        x = self.layer_norm(x)
        return x.transpose(1,2).contiguous()

class ResidualCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, dropout, n_feats):
        super(ResidualCNN, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels, out_channels, kernel,stride, padding=kernel//2)
        self.cnn2 = nn.Conv2d(out_channels, out_channels, kernel, stride, padding=kernel//2)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm1 = CNNLayerNorm(n_feats)
        self.layer_norm2 = CNNLayerNorm(n_feats)

    def forward(self, x):
        residual = x
        x = self.layer_norm1(x)
        x = F.gelu(x)
        x = self.dropout1(x)
        x = self.cnn1(x)
        x = self.layer_norm2(x)
        x = F.gelu(x)
        x = self.dropout2(x)
        x = self.cnn2(x)
        x+= residual
        return x 

class ResCNN(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample= None):
        super(ResCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1)
        )
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        output = F.max_pool2d(self.conv1(x),1)
        output = self.conv2(output)
        if self.downsample:
            residual = self.downsample(x)
        output += residual
        output = self.relu(output)
        return output

class BidiLGRU(nn.Module):
    def __init__(self, rnn_dim, hidden_size, dropout, batch_first):
        super(BidiLGRU, self).__init__()

        self.BiGRU = nn.GRU(
            input_size= rnn_dim,
            hidden_size= hidden_size,
            num_layers= 1,
            batch_first= batch_first,
            bidirectional = True
        )
        #self.layer_norm = nn.LayerNorm(rnn_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        #x = self.layer_norm(x)
        x = F.gelu(x)
        x, _ = self.BiGRU(x)
        x = self.dropout(x)
        return x

class NetRegressor(nn.Module):
    def __init__(self, rnn_dim, n_class, dropout):
        super(NetRegressor, self).__init__()
        self.hid1 = nn.Linear(rnn_dim*2, rnn_dim)
        self.hid2 = nn.Linear(rnn_dim, n_class)
        self.oupt = nn.Linear(n_class, 1)
        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.hid1.weight)
        nn.init.zeros_(self.hid1.bias)
        nn.init.xavier_uniform_(self.hid2.weight)
        nn.init.zeros_(self.hid2.bias)
        nn.init.xavier_uniform_(self.oupt.weight)
        nn.init.zeros_(self.oupt.bias)
    
    def forward(self, x):
        z = torch.relu(self.hid1(x))
        z = torch.relu(self.hid2(z))
        z = self.oupt(z) #No Activation
        return z

class FeaturesModel(nn.Module):
    def __init__(self, n_cnn_layers, n_rnn_layers, rnn_dim, n_class, n_feats, stride=2, dropout=0.1):
        super(FeaturesModel, self).__init__()
        n_feats = n_feats//2
        self.cnn = nn.Conv2d(500, 32, 1)
        
        self.rescnn_layers = nn.Sequential(*[
            ResCNN(
                32, 
                32
            ) for _ in range(n_cnn_layers)
        ])
        self.fully_connected = nn.Linear(n_feats, rnn_dim)
        self.birnn_layers = nn.Sequential(*[
            BidiLGRU(
                rnn_dim= rnn_dim if i==0 else rnn_dim*2,
                hidden_size= rnn_dim,
                dropout= dropout,
                batch_first=i==0
            ) for i in range(n_rnn_layers)
        ])
        
        self.classifier = nn.Sequential(
            nn.Linear(rnn_dim*2, rnn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(rnn_dim, n_class)
        )
    
    def forward(self,x):
        x = x.unsqueeze(0)
        x = x.transpose(1,2)
        x = self.cnn(x)
        #print("It has passed the first layer")
        #print("Size of x after entering: ",x.size())
        x = self.rescnn_layers(x)
        #print("ResNet done\tSize: ",x.size())
        x = x.squeeze(0)
        x = x.transpose(1,2)
        x = x.transpose(1,0)
        x = x[-1,:,:]
        x = x.transpose(1,0)
        #print("Size after messing up: ",x.size())
        x = self.fully_connected(x)
        #print("FC Done")
        x = self.birnn_layers(x)
        #print("RNN Done")
        #print("Size of the data going out the network:\t", x.size())
        #x = x[:,-1]
        #print("Size of the data to classify:\t", x.size())
        x = self.classifier(x)
        #print("Classifier done")
        #print("Data to the softmax:\t", x)
        return x
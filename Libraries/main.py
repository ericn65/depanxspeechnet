'''
JOINT LEARNING OF DEPRESION AND ANXIETY DIRECTLY FROM SPEECH SIGNALS

Author: Èric Quintana Aguasca

This is my first approach to it. By this point I will only try to compute PHQ-8, so just depression. 
'''
from data_Loader import dataLoaderSpec, dataCreator, dataLoaderPickle
from networks import FeaturesModel
from networks_v2 import CNNModel
from optimizer import train2, test2, optimizerNet
from optimizer import IterMeter

import os
import torch
import torch.nn as nn
import torchaudio as taudio
from comet_ml import Experiment

train_data, test_data = dataLoaderPickle('DATASET_NO_BALANCED_AT_aLL')

#Constants
learning_Rate = 0.01
batch_size = 128
epochs = 40
train_url = ""
test_url = ""
experiment = Experiment(api_key='dummy_key', disabled=True)

hparams = {
    "n_cnn_layers": 3,
    "n_rnn_layers": 3,
    "rnn_dim": 128,
    "n_class": 5,
    "n_feats": 64,
    "stride": 2,
    "dropout": 0.3,
    "learning_rate": learning_Rate,
    "batch_size": batch_size,
    "epochs": epochs
}

experiment.log_parameters(hparams)
use_cuda = torch.cuda.is_available()
torch.manual_seed(7)
device = torch.device("cuda" if use_cuda else "cpu")

#Building and training Model
model = FeaturesModel(
    hparams['n_cnn_layers'],
    hparams['n_rnn_layers'],
    hparams['rnn_dim'],
    hparams['n_class'],
    hparams['n_feats'],
    hparams['stride'],
    hparams['dropout']
).to(device).float()

print(model)
print('Number Model Parameters', sum([param.nelement() for param in model.parameters()]))

#Optimizing Model
optimizer, scheduler = optimizerNet(model, hparams, train_data)
#criterion = nn.NLLLoss().to(device)
weights = [1.0,1.0,0.5,1.0,1.0]
class_weights = torch.FloatTensor(weights)
criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)

iter_meter = IterMeter()
#for epoch in range(1, epochs + 1):
train2(model, device, train_data, criterion, optimizer, scheduler, epochs, iter_meter, experiment)
test2(model, device, test_data, criterion, optimizer, scheduler, epochs, iter_meter, experiment)


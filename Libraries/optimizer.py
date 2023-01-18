import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim 
import numpy as np


def optimizerNet(model, hparams):
    optimizer = optim.AdamW(
        model.parameters(),
        hparams['learning_rate']
    )    
    #features, labels = next(iter(data))
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=hparams['learning_rate'],
        steps_per_epoch= 100,
        epochs= hparams['epochs'],
        anneal_strategy= 'linear'
    )
    return optimizer, scheduler

class IterMeter(object):
    #Keeping track of iterations
    def __init__(self):
       self.val = 0

    def step(self):
        self.val += 1

    def get(self):
        return self.val 



import torch.nn.functional as F 
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sn 
import pandas as pd 
import matplotlib.pyplot as plt

def train(model, device, train_data, criterion, optimizer, scheduler, epochs, iter_meter, experiment):
    acc_list = []
    loss_list = []
    labels_list = []
    predicted_list = []
    loss_tot_list = []
    accuracy = []

    for epoch in range(epochs):
        for x, y in train_data:
            optimizer.zero_grad()
            z = model(x.float())

            labels_list.extend(y.detach().numpy())
            z_pred = F.log_softmax(z, dim=1)
            z_pred = (torch.max(torch.exp(z_pred), 1)[1])
            predicted_list.extend(z_pred.detach().numpy())
            y = y.long()
            #y = y.float()
            z = z.squeeze(0)
            #print("Label size: ", y.size())
            loss = criterion(z, y)
            loss.backward()
            optimizer.step()
            #scheduler.step()
            iter_meter.step()

            loss_list.append(loss.item())
            #Track accuracy
            total = y.size(0)
            _, predicted = torch.max(z.data, 1)
            correct = (predicted == y).sum().item()
            acc_list.append(correct/total)
            
        print('Train Epoch: {} \tLoss: {:.4f}\tAccuracy: {:.4f}'.format(
            epoch,
            np.mean(loss_list),
            np.mean(acc_list)
        ))
        loss_tot_list.append(np.mean(loss_list))
        accuracy.append(np.mean(acc_list))   
    
    #Printing Confusion matrix
    classes = ('0','1','2','3','4')
    cf_matrix = confusion_matrix(labels_list, predicted_list)
    df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix)*10, index = [i for i in classes], columns=[i for i in classes])
    plt.figure(figsize=(12,7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig('Confusion Matrix Training')
    
    #Printing training behaviour 
    fig, axs = plt.subplots(2)
    axs[0].plot(range(epochs), loss_tot_list)
    axs[0].set_title('Training Loss')
    axs[0].set(xlabel= 'Epoch',ylabel='Loss')
    axs[1].plot(range(epochs), accuracy)
    axs[1].set_title('Training Accuracy')
    axs[1].set(xlabel= 'Epoch',ylabel='Accuracy')
    plt.show()

def test(model, device, test_data, criterion, optimizer, scheduler, epochs, iter_meter, experiment):   
    acc_list = []
    loss_list = []
    labels_list = []
    predicted_list = []
    loss_tot_list = []
    accuracy = []

    for epoch in range(epochs):
        for x, y in test_data:
            optimizer.zero_grad()
            z = model(x.float())
            labels_list.extend(y.detach().numpy())
            z_pred = F.log_softmax(z, dim=1)
            z_pred = (torch.max(torch.exp(z_pred), 1)[1])
            predicted_list.extend(z_pred.detach().numpy())
            y = y.long()
            #y = y.float()
            z = z.squeeze(0)
            #print("Label size: ", y.size())
            loss = criterion(z, y)

            loss.backward()
            optimizer.step()
            #scheduler.step()
            iter_meter.step()

            loss_list.append(loss.item())
            #Track accuracy
            total = y.size(0)
            _, predicted = torch.max(z.data, 1)
            correct = (predicted == y).sum().item()
            acc_list.append(correct/total)
            
        print('Test Epoch: {} \tLoss: {:.4f}\tAccuracy: {:.4f}'.format(
            epoch,
            np.mean(loss_list),
            np.mean(acc_list)
        ))
        loss_tot_list.append(np.mean(loss_list))
        accuracy.append(np.mean(acc_list))    
    
    classes = ('0','1','2','3','4')
    cf_matrix = confusion_matrix(labels_list, predicted_list)
    df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix)*10, index = [i for i in classes], columns=[i for i in classes])
    plt.figure(figsize=(12,7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig('Confusion Matrix Test')

    fig, axs = plt.subplots(2)
    axs[0].plot(range(epochs), loss_tot_list)
    axs[0].set_title('Test Loss')
    axs[0].set(xlabel= 'Epoch',ylabel='Loss')
    axs[1].plot(range(epochs), accuracy)
    axs[1].set_title('Test Accuracy')
    axs[1].set(xlabel= 'Epoch',ylabel='Accuracy')
    plt.show()


def train2(model, device, train_data, criterion, optimizer, scheduler, epochs, iter_meter, experiment):
    acc_list = []
    loss_list = []
    labels_list = []
    predicted_list = []
    loss_tot_list = []
    accuracy = []

    for epoch in range(epochs):
        for x, y in train_data:
            optimizer.zero_grad()
            z = model(x.float())

            labels_list.extend(y.detach().numpy())
            z_pred = F.log_softmax(z, dim=1)
            z_pred = (torch.max(torch.exp(z_pred), 1)[1])
            predicted_list.extend(z_pred.detach().numpy())
            y = y.long()
            #y = y.float()
            z = z.squeeze(0)
            #print("Label size: ", y.size())
            loss = criterion(z, y)
            loss.backward()
            optimizer.step()
            #scheduler.step()
            iter_meter.step()

            loss_list.append(loss.item())
            #Track accuracy
            total = y.size(0)
            _, predicted = torch.max(z.data, 1)
            correct = (predicted == y).sum().item()
            acc_list.append(correct/total)
            
        print('Train Epoch: {} \tLoss: {:.4f}\tAccuracy: {:.4f}'.format(
            epoch,
            np.mean(loss_list),
            np.mean(acc_list)
        ))
        loss_tot_list.append(np.mean(loss_list))
        accuracy.append(np.mean(acc_list))   
    
    #Printing Confusion matrix
    classes = ('0','1','2','3')
    cf_matrix = confusion_matrix(labels_list, predicted_list)
    df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix)*10, index = [i for i in classes], columns=[i for i in classes])
    plt.figure(figsize=(12,7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig('Confusion Matrix Training')
    
    #Printing training behaviour 
    fig, axs = plt.subplots(2)
    axs[0].plot(range(epochs), loss_tot_list)
    axs[0].set_title('Training Loss')
    axs[0].set(xlabel= 'Epoch',ylabel='Loss')
    axs[1].plot(range(epochs), accuracy)
    axs[1].set_title('Training Accuracy')
    axs[1].set(xlabel= 'Epoch',ylabel='Accuracy')
    plt.show()

def test2(model, device, test_data, criterion, optimizer, scheduler, epochs, iter_meter, experiment):   
    acc_list = []
    loss_list = []
    labels_list = []
    predicted_list = []
    loss_tot_list = []
    accuracy = []

    for epoch in range(epochs):
        for x, y in test_data:
            optimizer.zero_grad()
            z = model(x.float())
            labels_list.extend(y.detach().numpy())
            z_pred = F.log_softmax(z, dim=1)
            z_pred = (torch.max(torch.exp(z_pred), 1)[1])
            predicted_list.extend(z_pred.detach().numpy())
            y = y.long()
            #y = y.float()
            z = z.squeeze(0)
            #print("Label size: ", y.size())
            loss = criterion(z, y)

            loss.backward()
            optimizer.step()
            #scheduler.step()
            iter_meter.step()

            loss_list.append(loss.item())
            #Track accuracy
            total = y.size(0)
            _, predicted = torch.max(z.data, 1)
            correct = (predicted == y).sum().item()
            acc_list.append(correct/total)
            
        print('Test Epoch: {} \tLoss: {:.4f}\tAccuracy: {:.4f}'.format(
            epoch,
            np.mean(loss_list),
            np.mean(acc_list)
        ))
        loss_tot_list.append(np.mean(loss_list))
        accuracy.append(np.mean(acc_list))    
    
    classes = ('0','1','2','3')
    cf_matrix = confusion_matrix(labels_list, predicted_list)
    df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix)*10, index = [i for i in classes], columns=[i for i in classes])
    plt.figure(figsize=(12,7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig('Confusion Matrix Test')

    fig, axs = plt.subplots(2)
    axs[0].plot(range(epochs), loss_tot_list)
    axs[0].set_title('Test Loss')
    axs[0].set(xlabel= 'Epoch',ylabel='Loss')
    axs[1].plot(range(epochs), accuracy)
    axs[1].set_title('Test Accuracy')
    axs[1].set(xlabel= 'Epoch',ylabel='Accuracy')
    plt.show()


def trainMLT(epochs, model_mlt, train_loader, optimizer, criterion_dep, criterion_anx, iter_meter):
    acc_dep_list = []
    acc_anx_list = []
    loss_list = []
    label_dep_list = []
    label_anx_list = []
    predicted_list_anx = []
    predicted_list_dep = []
    loss_tot_list = []
    accuracy_dep = []
    accuracy_anx = []

    #Training
    for epoch in range(epochs):
        model_mlt.train()
        total_training_loss = 0

        for x,y,z in train_loader:
            inputs = x
            label_dep = y.long()
            label_anx = z.long()
            label_dep_list.extend(label_dep.detach().numpy())
            label_anx_list.extend(label_anx.detach().numpy())

            optimizer.zero_grad()
            phq8, gad7 = model_mlt(inputs.float())
            phq8 = phq8.squeeze(0)
            gad7 = gad7.squeeze(0)

            predicted_list_dep.extend((torch.max(torch.exp(F.log_softmax(phq8,dim=1)),1)[1]).detach().numpy())
            predicted_list_anx.extend((torch.max(torch.exp(F.log_softmax(gad7,dim=1)),1)[1]).detach().numpy())

            loss_dep = criterion_dep(phq8,label_dep)
            loss_anx = criterion_anx(gad7, label_anx)

            loss = loss_dep + loss_anx
            loss.backward()
            optimizer.step()
            iter_meter.step()
            total_training_loss += loss

            loss_list.append(loss.item())
            #Track accuracy
            total_dep = label_dep.size(0)
            _, predicted = torch.max(phq8.data,1)
            correct_dep = (predicted == label_dep).sum().item()
            acc_dep_list.append(correct_dep/total_dep)  

            total_anx = label_anx.size(0)
            _, predicted = torch.max(gad7.data,1)
            correct_anx = (predicted==label_anx).sum().item()
            acc_anx_list.append(correct_anx/total_anx)

        print('Train Epoch: {} \tLoss: {:.4f}\tDepression Accuracy: {:.4f}\tAnxiety Accuracy: {:.4f}'.format(
            epoch,
            np.mean(loss_list),
            np.mean(acc_dep_list),
            np.mean(acc_anx_list)
        ))      
        loss_tot_list.append(np.mean(loss_list))
        accuracy_dep.append(np.mean(acc_dep_list))
        accuracy_anx.append(np.mean(acc_anx_list))
        
    #Printing Confusion Matrix
    classes_dep = ('0','1','2','3','4')
    cf_matrix_dep = confusion_matrix(label_dep_list, predicted_list_dep)
    df_cm = pd.DataFrame(cf_matrix_dep/np.sum(cf_matrix_dep)*10,index = [i for i in classes_dep], columns=[i for i in classes_dep])
    plt.figure(figsize=(12,7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig('Confusion Matrix Depression Training')

    classes_anx = ('0','1','2','3')
    cf_matrix_anx = confusion_matrix(label_anx_list, predicted_list_anx)
    df_cm = pd.DataFrame(cf_matrix_anx/np.sum(cf_matrix_anx)*10,index = [i for i in classes_anx], columns=[i for i in classes_anx])
    plt.figure(figsize=(12,7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig('Confusion Matrix Anxiety Training')

    fig, axs = plt.subplots(3)
    axs[0].plot(range(epochs), loss_tot_list)
    axs[0].set_title('Training Loss')
    axs[0].set(xlabel= 'Epoch', ylabel='Loss')
    axs[1].plot(range(epochs), accuracy_dep)
    axs[1].set_title('Training Depression Accuracy')
    axs[1].set(xlabel= 'Epoch', ylabel='Accuracy')
    axs[2].plot(range(epochs), accuracy_anx)
    axs[2].set_title('Training Anxiety Accuracy')
    axs[2].set(xlabel='Epoch', ylabel='Accuracy')
    plt.show()

def testMLT(epochs, model_mlt, test_loader, optimizer, criterion_dep, criterion_anx, iter_meter):
    #Test
    acc_dep_list = []
    acc_anx_list = []
    loss_list = []
    label_dep_list = []
    label_anx_list = []
    predicted_list_anx = []
    predicted_list_dep = []
    loss_tot_list = []
    accuracy_dep = []
    accuracy_anx = []
    for epoch in range(epochs):
        model_mlt.train()
        total_training_loss = 0

        for x,y,z in test_loader:
            inputs = x
            label_dep = y.long()
            label_anx = z.long()
            label_dep_list.extend(label_dep.detach().numpy())
            label_anx_list.extend(label_anx.detach().numpy())

            optimizer.zero_grad()
            phq8, gad7 = model_mlt(inputs.float())
            phq8 = phq8.squeeze(0)
            gad7 = gad7.squeeze(0)

            predicted_list_dep.extend((torch.max(torch.exp(F.log_softmax(phq8,dim=1)),1)[1]).detach().numpy())
            predicted_list_anx.extend((torch.max(torch.exp(F.log_softmax(gad7,dim=1)),1)[1]).detach().numpy())

            loss_dep = criterion_dep(phq8,label_dep)
            loss_anx = criterion_anx(gad7, label_anx)

            loss = loss_dep + loss_anx
            #loss.backward()
            #optimizer.step()
            iter_meter.step()
            total_training_loss += loss

            loss_list.append(loss.item())
            #Track accuracy
            total_dep = label_dep.size(0)
            _, predicted = torch.max(phq8.data,1)
            correct_dep = (predicted == label_dep).sum().item()
            acc_dep_list.append(correct_dep/total_dep)  

            total_anx = label_anx.size(0)
            _, predicted = torch.max(gad7.data,1)
            correct_anx = (predicted==label_anx).sum().item()
            acc_anx_list.append(correct_anx/total_anx)

        print('Test Epoch: {} \tLoss: {:.4f}\tDepression Accuracy: {:.4f}\tAnxiety Accuracy: {:.4f}'.format(
            epoch,
            np.mean(loss_list),
            np.mean(acc_dep_list),
            np.mean(acc_anx_list)
        ))      
        loss_tot_list.append(np.mean(loss_list))
        accuracy_dep.append(np.mean(acc_dep_list))
        accuracy_anx.append(np.mean(acc_anx_list))
        
    #Printing Confusion Matrix
    classes_dep = ('0','1','2','3','4')
    cf_matrix_dep = confusion_matrix(label_dep_list, predicted_list_dep)
    df_cm = pd.DataFrame(cf_matrix_dep/np.sum(cf_matrix_dep)*10,index = [i for i in classes_dep], columns=[i for i in classes_dep])
    plt.figure(figsize=(12,7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig('Confusion Matrix Depression Test')

    classes_anx = ('0','1','2','3')
    cf_matrix_anx = confusion_matrix(label_anx_list, predicted_list_anx)
    df_cm = pd.DataFrame(cf_matrix_anx/np.sum(cf_matrix_anx)*10,index = [i for i in classes_anx], columns=[i for i in classes_anx])
    plt.figure(figsize=(12,7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig('Confusion Matrix Anxiety Test')

    fig, axs = plt.subplots(3)
    axs[0].plot(range(epochs), loss_tot_list)
    axs[0].set_title('Test Loss')
    axs[0].set(xlabel= 'Epoch', ylabel='Loss')
    axs[1].plot(range(epochs), accuracy_dep)
    axs[1].set_title('Test Depression Accuracy')
    axs[1].set(xlabel= 'Epoch', ylabel='Accuracy')
    axs[2].plot(range(epochs), accuracy_anx)
    axs[2].set_title('Test Anxiety Accuracy')
    axs[2].set(xlabel='Epoch', ylabel='Accuracy')
    plt.show()   
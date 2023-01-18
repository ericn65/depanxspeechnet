'''
DATA LOADER AND TRANSFOMER

Author: Èric Quintana Aguasca

This is the new try for my project. This first script will be the data loader and transformer.
As we will use raw speech data as .wav files and we want them to be loaded as spectrograms.
We will also use data augmentation techniques over the spectrograms. 
This script must be used and changed depending on which data do we have. 
'''

import PIL
import librosa
from specAugment import spec_augment_pytorch
import torch 
import torch.nn as nn
import torchaudio as taudio
import glob
import os
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
from PIL import Image
import PIL.Image as I 
import torchvision as tvision 
import librosa.display
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import pickle
import math

def dataLoader(audio_path):
    data = []
    files = os.listdir(audio_path)
    for filename in glob.glob(os.path.join(audio_path, '*.wav')):
        #We must read all the folders, but they have really different names.
        audio, sampling_rate = librosa.load(filename)
        print(type(audio), type(sampling_rate))
        print(audio.shape, sampling_rate)
        data.append(audio)
    return data

def labelsLoader(category, task):
    os.chdir("C:/Users/ericq/OneDrive/Escriptori/TELECOM/MSC MATT/TFM/The research question/Analysing Features/Data_for_Eric/Data_for_Eric/Originals_changed")  
    if task == "Scripted":
        Feature_File = pd.read_excel('MDD_Praat-Scripted-RADAR-MDD-KCL-s1-18_04_2022_v3.xlsx')
    if task == "Unscripted":
        Feature_File = pd.read_excel('MDD_Praat-Unscripted-RADAR-MDD-KCL-s1-18_04_2022_v3.xlsx')
    
    Feature_File.insert(0, "Label", "NaN", True)

    if category == "PHQ8":
        Feature_File = Feature_File.drop(columns=['GAD7','IDS'])
        for index_FE, row_FE in Feature_File.iterrows():
            if Feature_File.loc[index_FE, category] <= 4:
                Feature_File.loc[index_FE, 'Label'] = "0"
            elif Feature_File.loc[index_FE, category] <= 9 and Feature_File.loc[index_FE, category] > 4:
                Feature_File.loc[index_FE, 'Label'] = "1"
            elif Feature_File.loc[index_FE, category] <= 14 and Feature_File.loc[index_FE, category] > 9:
                Feature_File.loc[index_FE, 'Label'] = "2"
            elif Feature_File.loc[index_FE, category] <= 19 and Feature_File.loc[index_FE, category] > 14:
                Feature_File.loc[index_FE, 'Label'] = "3"    
            elif Feature_File.loc[index_FE, category] <= 24 and Feature_File.loc[index_FE, category] > 19:
                Feature_File.loc[index_FE, 'Label'] = "4"
            else:
                Feature_File.loc[index_FE, 'Label'] = "NaN"
    elif category == "GAD7":            
        Feature_File = Feature_File.drop(columns=['PHQ8','IDS'])   
        for index_FE, row_FE in Feature_File.iterrows():
            if Feature_File.loc[index_FE, category] <= 4:
                Feature_File.loc[index_FE, 'Label'] = "0"
            elif Feature_File.loc[index_FE, category] <= 9 and Feature_File.loc[index_FE, category] > 4:
                Feature_File.loc[index_FE, 'Label'] = "1"
            elif Feature_File.loc[index_FE, category] <= 14 and Feature_File.loc[index_FE, category] > 9:
                Feature_File.loc[index_FE, 'Label'] = "2"
            elif Feature_File.loc[index_FE, category] > 14:
                Feature_File.loc[index_FE, 'Label'] = "3"     
    
    is_NaN = Feature_File.isnull()     
    row_has_NaN = is_NaN.any(axis=1)
    NaN_index =  row_has_NaN[row_has_NaN].index.values
    Feature_File.drop(NaN_index, inplace=True)

    delete_rows = []
    for index_FE, row_FE in Feature_File.iterrows():
        if Feature_File.loc[index_FE,'Duration'] < 2:
            delete_rows.append(index_FE)
    Feature_File = Feature_File.drop(delete_rows) 
    Feature_File = Feature_File.reset_index()

    return Feature_File

def labelsMLTLoader(task):
    os.chdir("C:/Users/ericq/OneDrive/Escriptori/TELECOM/MSC MATT/TFM/The research question/Analysing Features/Data_for_Eric/Data_for_Eric/Originals_changed")  
    if task == "Scripted":
        Feature_File_1 = pd.read_excel('MDD_Praat-Scripted-RADAR-MDD-CIBER-s1-18_04_2022_v3.xlsx')
        Feature_File_2 = pd.read_excel('MDD_Praat-Scripted-RADAR-MDD-IISPV-s1-18_04_2022_v3.xlsx')
        Feature_File = pd.concat([Feature_File_1, Feature_File_2])
        Feature_File = Feature_File.reset_index()
        Feature_File = Feature_File.drop(columns=['index'])
    
    if task == "Unscripted":
        Feature_File_1 = pd.read_excel('MDD_Praat-Unscripted-RADAR-MDD-CIBER-s1-18_04_2022_v3.xlsx')
        Feature_File_2 = pd.read_excel('MDD_Praat-Unscripted-RADAR-MDD-IISPV-s1-18_04_2022_v3.xlsx')
        Feature_File = pd.concat([Feature_File_1, Feature_File_2])
        Feature_File = Feature_File.reset_index()
        Feature_File = Feature_File.drop(columns=['index'])

    
    Feature_File.insert(0, "LabelDep", "NaN", True)
    Feature_File.insert(0, "LabelAnx", "NaN", True)

    category = "PHQ8"
    for index_FE, row_FE in Feature_File.iterrows():
        if Feature_File.loc[index_FE, category] <= 4:
            Feature_File.loc[index_FE, 'LabelDep'] = "0"
        elif Feature_File.loc[index_FE, category] <= 9 and Feature_File.loc[index_FE, category] > 4:
            Feature_File.loc[index_FE, 'LabelDep'] = "1"
        elif Feature_File.loc[index_FE, category] <= 14 and Feature_File.loc[index_FE, category] > 9:
            Feature_File.loc[index_FE, 'LabelDep'] = "2"
        elif Feature_File.loc[index_FE, category] <= 19 and Feature_File.loc[index_FE, category] > 14:
            Feature_File.loc[index_FE, 'LabelDep'] = "3"    
        elif Feature_File.loc[index_FE, category] <= 24 and Feature_File.loc[index_FE, category] > 19:
            Feature_File.loc[index_FE, 'LabelDep'] = "4"
        else:
            Feature_File.loc[index_FE, 'LabelDep'] = "NaN"
         
    category = "GAD7"
    for index_FE, row_FE in Feature_File.iterrows():
        if Feature_File.loc[index_FE, category] <= 4:
            Feature_File.loc[index_FE, 'LabelAnx'] = "0"
        elif Feature_File.loc[index_FE, category] <= 9 and Feature_File.loc[index_FE, category] > 4:
            Feature_File.loc[index_FE, 'LabelAnx'] = "1"
        elif Feature_File.loc[index_FE, category] <= 14 and Feature_File.loc[index_FE, category] > 9:
            Feature_File.loc[index_FE, 'LabelAnx'] = "2"
        elif Feature_File.loc[index_FE, category] > 14:
            Feature_File.loc[index_FE, 'LabelAnx'] = "3"    

    is_NaN = Feature_File.isnull()     
    row_has_NaN = is_NaN.any(axis=1)
    NaN_index =  row_has_NaN[row_has_NaN].index.values
    Feature_File.drop(NaN_index, inplace=True)

    delete_rows = []
    for index_FE, row_FE in Feature_File.iterrows():
        if Feature_File.loc[index_FE,'Duration'] < 2:
            delete_rows.append(index_FE)
    Feature_File = Feature_File.drop(delete_rows) 
    Feature_File = Feature_File.reset_index()

    return Feature_File
    

def dataLoaderSpec(spec_path, categroy):
    spectrograms = []
    labels = []
    name = []
    name = [x[0] for x in os.walk(spec_path)]
    name.pop(0)
    script_label = labelsLoader(categroy, "Scripted")
    unscript_label = labelsLoader(categroy, "Unscripted")
    for names in range(len(name)):    
        File_path = name[names]
        files = os.listdir(File_path)
        for filename in glob.glob(os.path.join(File_path, '*.png')):
            spec = I.open(filename)
            to_tensor = tvision.transforms.ToTensor()
            to_tensor(spec).squeeze(0).transpose(0,1)
            
            x = File_path.split("\\")
            id = x[14]
            y = filename.split("_")
            date = y[3]
            y = date.split("\\")
            date = y[1]
            z = filename.split("-")
            task = z[11]
            if task == "scripted":
                for index_FE, row_FE in script_label.iterrows():
                    if id == script_label.loc[index_FE, 'participant_ID']:
                        label_date = str(script_label.loc[index_FE, 'Recording_Date']).split("-")
                        label_date = label_date[0] + label_date[1] + label_date[2]
                        label_date = str(label_date).split(" ")
                        label_date = label_date[0]
                        if date == label_date:
                            labels.append(script_label.loc[index_FE, 'Label'])
                            spectrograms.append(to_tensor(spec).squeeze(0).transpose(0,1))
                            print("spect appended")

            elif task == "unscripted":
                for index_FE, row_FE in unscript_label.iterrows():
                    if id == unscript_label.loc[index_FE, 'participant_ID']:
                        label_date = str(script_label.loc[index_FE, 'Recording_Date']).split("-")
                        label_date = label_date[0] + label_date[1] + label_date[2]
                        label_date = str(label_date).split(" ")
                        label_date = label_date[0]
                        if date == label_date:
                            labels.append(unscript_label.loc[index_FE, 'Label'])
                            spectrograms.append(to_tensor(spec).squeeze(0).transpose(0,1))
                            print("Spect appended")
            else:
                print("Not Working")
    

        '''
    spectrograms = nn.utils.rnn.pad_sequence(
        spectrograms,
        batch_first= True  
    ).unsqueeze(1).transpose(2,3)  

    labels = nn.utils.rnn.pad_sequence(
        labels,
        batch_first= True
    )

    input_lengths = len(spectrograms)
    label_lengths = len(labels) '''

    labels = np.array([labels])
    labels = labels.astype('float').reshape(-1, 1)
    labels = torch.from_numpy(labels)
    spect_train, spect_val, label_train, label_val = dataSplit(spectrograms, labels)
    input_lengths = len(spectrograms)
    label_lengths = len(labels) 
    
    return spect_train, spect_val, label_train, label_val, input_lengths, label_lengths

def datatoSpec(data, sr, path):
    spectrograms = []
    for waves in range(len(data)):
        spec = librosa.stft(data[waves])
        spec = librosa.amplitude_to_db(abs(spec))
        spectrograms.append(spec)
        '''plt.figure(figsize=(14,5))
        librosa.display.specshow(spec, sr=sr, x_axis='time', y_axis='hz')
        plt.colorbar()'''
        image = spec.to_image(shape=(224, 224),invert=True)
        image.save(path)
    return spectrograms

def dataSplit(spectrograms, labels):
    spectrograms, labels = shuffle(spectrograms, labels, random_state=0)
    X_train, x_val, y_train, y_val = train_test_split(spectrograms, labels, test_size=0.3)
    return X_train, x_val, y_train, y_val

def dataLoadertoSpecLabels(train_loader, category):
    name =[]
    data = []
    audio = []
    spectrograms = []
    labels = []
    input_lengths = []
    label_lengths = []
    name = [x[0] for x in os.walk(train_loader)]
    name.pop(0)

    path = train_loader + "\\Spectrograms_RADAR_KCL"
    i = 0
    SAMPLE_RATE = 16000
    N_FFT = 1024
    HOP_LENGTH = 512
    N_MELS = 128
    F_MAX = 8000

    script_label = labelsLoader(category, "Scripted")
    unscript_label = labelsLoader(category, "Unscripted")

    for names in range(len(name)):
        file_path = name[names]
        files = os.listdir(file_path)
        for filename in glob.glob(os.path.join(file_path, '*.wav')):
            #We must read all the folders, but they have really different names.
            audio, sampling_rate = librosa.load(filename, sr=SAMPLE_RATE)
            try:
                os.chdir(path)
            except:
                os.mkdir(path)
                os.chdir(path)
            spec= librosa.feature.melspectrogram(
                y= audio, 
                sr=SAMPLE_RATE, 
                n_fft=N_FFT, 
                hop_length=HOP_LENGTH, 
                n_mels=N_MELS, 
                fmax = F_MAX
            )
            print(spec.size)
            spec = librosa.power_to_db(spec)
            x = file_path.split("\\")
            id = x[14]
            y = filename.split("_")
            date = y[3]
            y = date.split("\\")
            date = y[1]
            z = filename.split("-")
            task = z[11]
            if task == "scripted":
                for index_FE, row_FE in script_label.iterrows():
                    if id == script_label.loc[index_FE, 'participant_ID']:
                        label_date = str(script_label.loc[index_FE, 'Recording_Date']).split("-")
                        label_date = label_date[0] + label_date[1] + label_date[2]
                        label_date = str(label_date).split(" ")
                        label_date = label_date[0]
                        if date == label_date:
                            labels.append(script_label.loc[index_FE, 'Label'])
                            spectrograms.append(spec)

            elif task == "unscripted":
                for index_FE, row_FE in unscript_label.iterrows():
                    if id == unscript_label.loc[index_FE, 'participant_ID']:
                        label_date = str(script_label.loc[index_FE, 'Recording_Date']).split("-")
                        label_date = label_date[0] + label_date[1] + label_date[2]
                        label_date = str(label_date).split(" ")
                        label_date = label_date[0]
                        if date == label_date:
                            labels.append(unscript_label.loc[index_FE, 'Label'])
                            spectrograms.append(spec)
            else:
                print("Not Working")
    
    labels = np.array([labels])
    labels = labels.astype('float').reshape(-1, 1)
    labels = torch.from_numpy(labels)
    spect_train, spect_val, label_train, label_val = dataSplit(spectrograms, labels)
    input_lengths = len(spectrograms)
    label_lengths = len(labels) 
    
    return spect_train, spect_val, label_train, label_val, input_lengths, label_lengths

def dataTransform(audio_path):
    audio, sampling_rate = librosa.load(audio_path)
    mel_spectrogram = librosa.feature.melspectrogram(
        y= audio,
        sr= sampling_rate,
        n_mels= 256,
        hop_lentgth= 128,
        fmax= 8000
    )
    return mel_spectrogram


def dataAugment(mel_spectrogram):
    # It should be used in a for loop to get all the possible augmentations.
    wraped_masked_spectrogram = spec_augment_pytorch.spec_augment(mel_spectrogram=mel_spectrogram)
    return wraped_masked_spectrogram

def data_processing(data, data_type="train"):
    spectrograms = []
    labels = []
    input_lengths = []
    label_lengths = []
    
    train_audio_transforms = nn.Sequential(
        taudio.transforms.MelSpectrogram(
            sample_rate= 16000,
            n_mels= 256
        ),
        taudio.transforms.FrequencyMasking(freq_mask_param=15),
        taudio.transforms.TimeMasking(time_mask_param=35)
    )
    valid_audio_transforms = taudio.transforms.MelSpectrogram()
    
    for waveform in data:
        if data_type == 'train':
            spec = train_audio_transforms(waveform).squeeze(0).transpose(0,1)
        else: 
            spec = valid_audio_transforms(waveform).squeeze(0).transpose(0,1)
        
        spectrograms.append(spec)
        #Now we should do something with labeling :/
        input_lengths.append(spec.shape[0]//2)

    spectrograms = nn.utils.rnn.pad_sequence(
        spectrograms,
        batch_first= True  
    ).unsqueeze(1).transpose(2,3)
    #Something else with labels

    return spectrograms, input_lengths

def dataSaver(dataset, filename):
    #filename = 'Scripted_PHQ8_Eng'
    outfile = open(filename, 'wb')
    pickle.dump(dataset, outfile)
    outfile.close()

def justDatasetLoaderPickle(filename):
    infile = open(filename, 'rb')
    dataset = pickle.load(infile)
    infile.close()
    return dataset


def dataLoaderPickle(filename):
    infile = open(filename, 'rb')
    dataset = pickle.load(infile)
    infile.close()
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])
    print("Training size:", len(train_set))
    print("Testing size:",len(test_set))

    train_dataloader = torch.utils.data.DataLoader(
        train_set,
        batch_size=128,
        num_workers=2,
        shuffle=True
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_set,
        batch_size=128,
        num_workers=2,
        shuffle=True
    )

    return train_dataloader, test_dataloader

def anxietyDataCreator(train_loader, category, pickleName):
    name =[]
    audio = []
    #spectrograms = np.array()
    labels = []

    name = [x[0] for x in os.walk(train_loader)]
    name.pop(0)

    #path =   + "\\Spectrograms_RADAR_KCL"
    i = 0
    SAMPLE_RATE = 16000
    N_FFT = 512
    HOP_LENGTH = int(0.01*SAMPLE_RATE)
    N_MELS = 40
    F_MAX = 8000
    WIN_LENGTH= int(0.025*SAMPLE_RATE)
    SEQUENCE_LENGTH= 500

    script_label = labelsLoader(category, "Scripted")
    unscript_label = labelsLoader(category, "Unscripted")
    spectrograms = np.zeros((1,SEQUENCE_LENGTH,N_MELS))
    control = True

    for i, names in enumerate(name):
        #file_path = name[names]
        file_path = names
        files = os.listdir(file_path)
        count = 0

        for j, filename in enumerate(glob.glob(os.path.join(file_path, '*.wav'))):
            #We must read all the folders, but they have really different names.
            x = file_path.split("\\")
            id = x[14]
            y = filename.split("_")
            date = y[3]
            y = date.split("\\")
            date = y[1]
            z = filename.split("-")
            task = z[11]
            if task == "scripted":
                count+=1
                print("Signal processed "+ str(count) + "_" + str(i))
                audio, sampling_rate = librosa.load(filename, sr=SAMPLE_RATE)
                spec= librosa.feature.melspectrogram(
                    y= audio, 
                    sr=SAMPLE_RATE, 
                    n_fft=N_FFT, 
                    hop_length=HOP_LENGTH, 
                    n_mels=N_MELS, 
                    fmax = F_MAX,
                    win_length=WIN_LENGTH
                )
                spec = librosa.power_to_db(spec).transpose()
                #print(spec.shape, "Spectrogram shape")
                
                div = spec.shape[0] // SEQUENCE_LENGTH
                size_new = int(div * SEQUENCE_LENGTH)
                spec = spec[:size_new]
                spec = spec.reshape((div, SEQUENCE_LENGTH, N_MELS))
                print("New Spectrogram")
                for index_FE, row_FE in script_label.iterrows():
                    if id == script_label.loc[index_FE, 'participant_ID']:
                        label_date = str(script_label.loc[index_FE, 'Recording_Date']).split("-")
                        label_date = label_date[0] + label_date[1] + label_date[2]
                        label_date = str(label_date).split(" ")
                        label_date = label_date[0]
                        if date == label_date:
                            print("The new spectrogram has a target")
                            for k in range(div):
                                lab_to_add = script_label.loc[index_FE, 'Label']
                                lab = script_label.loc[index_FE, 'Label']
                                labels.append(lab_to_add)
                                print("Label: " + script_label.loc[index_FE, 'Label'] + "\tID: " + script_label.loc[index_FE, 'participant_ID'] + "\tOriginal label: " + str(script_label.loc[index_FE, category]))
                            spectrograms = np.concatenate((spectrograms, spec))
                            print(spec.shape, "Spectrogram shape")
                
                if int(lab_to_add) > 1:
                    y_noise = audio
                    rms = math.sqrt(np.mean(y_noise**2))
                    noise = np.random.normal(0, rms, y_noise.shape[0])
                    y_noise = y_noise + noise                               
                    spec_t= librosa.feature.melspectrogram(
                        y= y_noise, 
                        sr=SAMPLE_RATE, 
                        n_fft=N_FFT, 
                        hop_length=HOP_LENGTH, 
                        n_mels=N_MELS, 
                        fmax = F_MAX,
                        win_length=WIN_LENGTH
                    )
                    spec_t = librosa.power_to_db(spec_t).transpose()
                    #print(spec_t.shape, "Spectrogram shape NOISE")

                    div2 = spec_t.shape[0] // SEQUENCE_LENGTH
                    size_new = int(div2 * SEQUENCE_LENGTH)
                    spec_t = spec_t[:size_new]
                    spec_t = spec_t.reshape((div2, SEQUENCE_LENGTH, N_MELS))
                    
                    print("New Spectrogram augmented")
                    for index_FE, row_FE in script_label.iterrows():
                        if id == script_label.loc[index_FE, 'participant_ID']:
                            label_date = str(script_label.loc[index_FE, 'Recording_Date']).split("-")
                            label_date = label_date[0] + label_date[1] + label_date[2]
                            label_date = str(label_date).split(" ")
                            label_date = label_date[0]
                            if date == label_date:
                                print("The new spectrogram has a target")
                                for k in range(div):
                                    lab_to_add = script_label.loc[index_FE, 'Label']
                                    labels.append(lab_to_add)
                                    print("Label: " + script_label.loc[index_FE, 'Label'] + "\tID: " + script_label.loc[index_FE, 'participant_ID'] + "\tOriginal label: " + str(script_label.loc[index_FE, category]))
                                spectrograms = np.concatenate((spectrograms, spec_t))
                                print(spec.shape, "Spectrogram shape NOISE")
                    
                    print("DATA AUGMENTED")      

                if int(lab_to_add) > 2:
                    y_noise = audio
                    rms = math.sqrt(np.mean(y_noise**2))
                    noise = np.random.normal(0, rms, y_noise.shape[0])
                    y_noise = y_noise + noise                               
                    spec_t= librosa.feature.melspectrogram(
                        y= y_noise, 
                        sr=SAMPLE_RATE, 
                        n_fft=N_FFT, 
                        hop_length=HOP_LENGTH, 
                        n_mels=N_MELS, 
                        fmax = F_MAX,
                        win_length=WIN_LENGTH
                    )
                    spec_t = librosa.power_to_db(spec_t).transpose()
                    #print(spec_t.shape, "Spectrogram shape NOISE")

                    div2 = spec_t.shape[0] // SEQUENCE_LENGTH
                    size_new = int(div2 * SEQUENCE_LENGTH)
                    spec_t = spec_t[:size_new]
                    spec_t = spec_t.reshape((div2, SEQUENCE_LENGTH, N_MELS))
                    
                    print("New Spectrogram augmented")
                    for index_FE, row_FE in script_label.iterrows():
                        if id == script_label.loc[index_FE, 'participant_ID']:
                            label_date = str(script_label.loc[index_FE, 'Recording_Date']).split("-")
                            label_date = label_date[0] + label_date[1] + label_date[2]
                            label_date = str(label_date).split(" ")
                            label_date = label_date[0]
                            if date == label_date:
                                print("The new spectrogram has a target")
                                for k in range(div):
                                    lab_to_add = script_label.loc[index_FE, 'Label']
                                    labels.append(lab_to_add)
                                    print("Label: " + script_label.loc[index_FE, 'Label'] + "\tID: " + script_label.loc[index_FE, 'participant_ID'] + "\tOriginal label: " + str(script_label.loc[index_FE, category]))
                                spectrograms = np.concatenate((spectrograms, spec_t))
                                print(spec.shape, "Spectrogram shape NOISE")
                    
                    print("DATA AUGMENTED")    
                print("Length Spectrograms: ",spectrograms.shape)
                print("Length labels: ", len(labels))

    spectrograms = spectrograms[1:,:,:]
    print(spectrograms.shape, "Spectrograms array")
    print(len(labels))

    labels = np.asarray(labels, dtype=int)
    dataset = TensorDataset(
        torch.from_numpy(spectrograms),
        torch.from_numpy(labels)
    )

    os.chdir("C:\\Users\\ericq\\OneDrive\\Escriptori\\TELECOM\\MSC MATT\\TFM\\The research question\\The Solution\\CNN+RNN")
    dataSaver(dataset, pickleName)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])
    print("Training size:", len(train_set))
    print("Testing size:",len(test_set))

    
    #Print histogram of classes
    n, bins, patches = plt.hist(x=labels,bins='auto',color='#0504aa',alpha=0.7, rwidth=0.85)
    plt.grid(axis='y',alpha=0.75)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Is my data balanced?')
    plt.text(23,45,r'$\mu15, b=3$')
    maxfreq = n.max()
    plt.ylim(ymax=np.ceil(maxfreq/10)*10 if maxfreq % 10 else maxfreq + 10)
    plt.show()

    train_dataloader = torch.utils.data.DataLoader(
        train_set,
        batch_size=128,
        num_workers=2,
        shuffle=True
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_set,
        batch_size=128,
        num_workers=2,
        shuffle=True
    )

    return train_dataloader, test_dataloader, dataset


def dataCreator(train_loader, category, pickleName):
    name =[]
    audio = []
    #spectrograms = np.array()
    labels = []

    name = [x[0] for x in os.walk(train_loader)]
    name.pop(0)

    path = train_loader + "\\Spectrograms_RADAR_KCL"
    i = 0
    SAMPLE_RATE = 16000
    N_FFT = 512
    HOP_LENGTH = int(0.01*SAMPLE_RATE)
    N_MELS = 40
    F_MAX = 8000
    WIN_LENGTH= int(0.025*SAMPLE_RATE)
    SEQUENCE_LENGTH= 500

    script_label = labelsLoader(category, "Scripted")
    unscript_label = labelsLoader(category, "Unscripted")
    spectrograms = np.zeros((1,SEQUENCE_LENGTH,N_MELS))
    control = True

    for i, names in enumerate(name):
        #file_path = name[names]
        file_path = names
        files = os.listdir(file_path)
        count = 0

        for j, filename in enumerate(glob.glob(os.path.join(file_path, '*.wav'))):
            #We must read all the folders, but they have really different names.
            x = file_path.split("\\")
            id = x[14]
            y = filename.split("_")
            date = y[3]
            y = date.split("\\")
            date = y[1]
            z = filename.split("-")
            task = z[11]
            if task == "scripted":
                count+=1
                print("Signal processed "+ str(count) + "_" + str(i))
                '''
                if count >= 128:
                    break'''
                audio, sampling_rate = librosa.load(filename, sr=SAMPLE_RATE)
                spec= librosa.feature.melspectrogram(
                    y= audio, 
                    sr=SAMPLE_RATE, 
                    n_fft=N_FFT, 
                    hop_length=HOP_LENGTH, 
                    n_mels=N_MELS, 
                    fmax = F_MAX,
                    win_length=WIN_LENGTH
                )
                spec = librosa.power_to_db(spec).transpose()
                #print(spec.shape, "Spectrogram shape")
                
                div = spec.shape[0] // SEQUENCE_LENGTH
                size_new = int(div * SEQUENCE_LENGTH)
                spec = spec[:size_new]
                spec = spec.reshape((div, SEQUENCE_LENGTH, N_MELS))
            
                print("New Spectrogram")
                for index_FE, row_FE in script_label.iterrows():
                    if id == script_label.loc[index_FE, 'participant_ID']:
                        label_date = str(script_label.loc[index_FE, 'Recording_Date']).split("-")
                        label_date = label_date[0] + label_date[1] + label_date[2]
                        label_date = str(label_date).split(" ")
                        label_date = label_date[0]
                        if date == label_date:
                            print("The new spectrogram has a target")
                            for k in range(div):
                                lab_to_add = script_label.loc[index_FE, category]
                                lab = script_label.loc[index_FE, 'Label']
                                if int(script_label.loc[index_FE, 'Label']) == 2:
                                    if control == True:
                                        control = False
                                    else:
                                        control = True
                                labels.append(lab_to_add)
                                print("Label: " + script_label.loc[index_FE, 'Label'] + "\tID: " + script_label.loc[index_FE, 'participant_ID'] + "\tOriginal label: " + str(script_label.loc[index_FE, category]))
                            spectrograms = np.concatenate((spectrograms, spec))
                            print(spec.shape, "Spectrogram shape")
                        
                if int(lab_to_add) > 14:
                    y_noise = audio
                    rms = math.sqrt(np.mean(y_noise**2))
                    noise = np.random.normal(0, rms, y_noise.shape[0])
                    y_noise = y_noise + noise                               
                    spec_t= librosa.feature.melspectrogram(
                        y= y_noise, 
                        sr=SAMPLE_RATE, 
                        n_fft=N_FFT, 
                        hop_length=HOP_LENGTH, 
                        n_mels=N_MELS, 
                        fmax = F_MAX,
                        win_length=WIN_LENGTH
                    )
                    spec_t = librosa.power_to_db(spec_t).transpose()
                    #print(spec_t.shape, "Spectrogram shape NOISE")

                    div2 = spec_t.shape[0] // SEQUENCE_LENGTH
                    size_new = int(div2 * SEQUENCE_LENGTH)
                    spec_t = spec_t[:size_new]
                    spec_t = spec_t.reshape((div2, SEQUENCE_LENGTH, N_MELS))
                    
                    print("New Spectrogram augmented")
                    for index_FE, row_FE in script_label.iterrows():
                        if id == script_label.loc[index_FE, 'participant_ID']:
                            label_date = str(script_label.loc[index_FE, 'Recording_Date']).split("-")
                            label_date = label_date[0] + label_date[1] + label_date[2]
                            label_date = str(label_date).split(" ")
                            label_date = label_date[0]
                            if date == label_date:
                                print("The new spectrogram has a target")
                                for k in range(div):
                                    lab_to_add = script_label.loc[index_FE, category]
                                    labels.append(lab_to_add)
                                    print("Label: " + script_label.loc[index_FE, 'Label'] + "\tID: " + script_label.loc[index_FE, 'participant_ID'] + "\tOriginal label: " + str(script_label.loc[index_FE, category]))
                                spectrograms = np.concatenate((spectrograms, spec_t))
                                print(spec.shape, "Spectrogram shape NOISE")
                    
                    print("DATA AUGMENTED")      

                if int(lab_to_add) > 19:
                    y_noise = audio
                    rms = math.sqrt(np.mean(y_noise**2))
                    noise = np.random.normal(0, rms, y_noise.shape[0])
                    y_noise = y_noise + noise                               
                    spec_t= librosa.feature.melspectrogram(
                        y= y_noise, 
                        sr=SAMPLE_RATE, 
                        n_fft=N_FFT, 
                        hop_length=HOP_LENGTH, 
                        n_mels=N_MELS, 
                        fmax = F_MAX,
                        win_length=WIN_LENGTH
                    )
                    spec_t = librosa.power_to_db(spec_t).transpose()
                    #print(spec_t.shape, "Spectrogram shape NOISE")

                    div2 = spec_t.shape[0] // SEQUENCE_LENGTH
                    size_new = int(div2 * SEQUENCE_LENGTH)
                    spec_t = spec_t[:size_new]
                    spec_t = spec_t.reshape((div2, SEQUENCE_LENGTH, N_MELS))
                    
                    print("New Spectrogram augmented")
                    for index_FE, row_FE in script_label.iterrows():
                        if id == script_label.loc[index_FE, 'participant_ID']:
                            label_date = str(script_label.loc[index_FE, 'Recording_Date']).split("-")
                            label_date = label_date[0] + label_date[1] + label_date[2]
                            label_date = str(label_date).split(" ")
                            label_date = label_date[0]
                            if date == label_date:
                                print("The new spectrogram has a target")
                                for k in range(div):
                                    lab_to_add = script_label.loc[index_FE, category]
                                    labels.append(lab_to_add)
                                    print("Label: " + script_label.loc[index_FE, 'Label'] + "\tID: " + script_label.loc[index_FE, 'participant_ID'] + "\tOriginal label: " + str(script_label.loc[index_FE, category]))
                                spectrograms = np.concatenate((spectrograms, spec_t))
                                print(spec.shape, "Spectrogram shape NOISE")
                    
                    print("DATA AUGMENTED")    
                
                if (int(lab) == 2) and (control == True):
                    y_noise = audio
                    rms = math.sqrt(np.mean(y_noise**2))
                    noise = np.random.normal(0, rms, y_noise.shape[0])
                    y_noise = y_noise + noise                               
                    spec_t= librosa.feature.melspectrogram(
                        y= y_noise, 
                        sr=SAMPLE_RATE, 
                        n_fft=N_FFT, 
                        hop_length=HOP_LENGTH, 
                        n_mels=N_MELS, 
                        fmax = F_MAX,
                        win_length=WIN_LENGTH
                    )
                    spec_t = librosa.power_to_db(spec_t).transpose()
                    #print(spec_t.shape, "Spectrogram shape NOISE")

                    div2 = spec_t.shape[0] // SEQUENCE_LENGTH
                    size_new = int(div2 * SEQUENCE_LENGTH)
                    spec_t = spec_t[:size_new]
                    spec_t = spec_t.reshape((div2, SEQUENCE_LENGTH, N_MELS))
                    
                    print("New Spectrogram augmented")
                    for index_FE, row_FE in script_label.iterrows():
                        if id == script_label.loc[index_FE, 'participant_ID']:
                            label_date = str(script_label.loc[index_FE, 'Recording_Date']).split("-")
                            label_date = label_date[0] + label_date[1] + label_date[2]
                            label_date = str(label_date).split(" ")
                            label_date = label_date[0]
                            if date == label_date:
                                print("The new spectrogram has a target")
                                for k in range(div):
                                    lab_to_add = script_label.loc[index_FE, category]
                                    ind = index_FE
                                    labels.append(lab_to_add)
                                    print("Label: " + script_label.loc[index_FE, 'Label'] + "\tID: " + script_label.loc[index_FE, 'participant_ID'] + "\tOriginal label: " + str(script_label.loc[index_FE, category]))
                                spectrograms = np.concatenate((spectrograms, spec_t))
                                print(spec.shape, "Spectrogram shape NOISE")
                    
                    print("DATA AUGMENTED")

                print("Length Spectrograms: ",spectrograms.shape)
                print("Length labels: ", len(labels))


            
            elif task == "unscripted":
                count+=1
                print("Signal processed "+ str(count) + "_" + str(i))
                '''
                if count >= 128:
                    break'''
                audio, sampling_rate = librosa.load(filename, sr=SAMPLE_RATE)
                spec= librosa.feature.melspectrogram(
                    y= audio, 
                    sr=SAMPLE_RATE, 
                    n_fft=N_FFT, 
                    hop_length=HOP_LENGTH, 
                    n_mels=N_MELS, 
                    fmax = F_MAX,
                    win_length=WIN_LENGTH
                )
                spec = librosa.power_to_db(spec).transpose()
                #print(spec.shape, "Spectrogram shape")
                
                div = spec.shape[0] // SEQUENCE_LENGTH
                size_new = int(div * SEQUENCE_LENGTH)
                spec = spec[:size_new]
                spec = spec.reshape((div, SEQUENCE_LENGTH, N_MELS))
            
                print("New Spectrogram")
                for index_FE, row_FE in unscript_label.iterrows():
                    if id == unscript_label.loc[index_FE, 'participant_ID']:
                        label_date = str(unscript_label.loc[index_FE, 'Recording_Date']).split("-")
                        label_date = label_date[0] + label_date[1] + label_date[2]
                        label_date = str(label_date).split(" ")
                        label_date = label_date[0]
                        if date == label_date:
                            print("The new spectrogram has a target")
                            for k in range(div):
                                lab_to_add = unscript_label.loc[index_FE, category]
                                lab = unscript_label.loc[index_FE, 'Label']
                                if int(script_label.loc[index_FE, 'Label']) == 2:
                                    if control == True:
                                        control = False
                                    else:
                                        control = True
                                labels.append(lab_to_add)
                            spectrograms = np.concatenate((spectrograms, spec))
                            print(spec.shape, "Spectrogram shape")
                        
                if int(lab_to_add) > 14:
                    y_noise = audio
                    rms = math.sqrt(np.mean(y_noise**2))
                    noise = np.random.normal(0, rms, y_noise.shape[0])
                    y_noise = y_noise + noise                               
                    spec_t= librosa.feature.melspectrogram(
                        y= y_noise, 
                        sr=SAMPLE_RATE, 
                        n_fft=N_FFT, 
                        hop_length=HOP_LENGTH, 
                        n_mels=N_MELS, 
                        fmax = F_MAX,
                        win_length=WIN_LENGTH
                    )
                    spec_t = librosa.power_to_db(spec_t).transpose()
                    #print(spec_t.shape, "Spectrogram shape NOISE")

                    div2 = spec_t.shape[0] // SEQUENCE_LENGTH
                    size_new = int(div2 * SEQUENCE_LENGTH)
                    spec_t = spec_t[:size_new]
                    spec_t = spec_t.reshape((div2, SEQUENCE_LENGTH, N_MELS))
                    
                    print("New Spectrogram augmented")
                    for index_FE, row_FE in unscript_label.iterrows():
                        if id == unscript_label.loc[index_FE, 'participant_ID']:
                            label_date = str(unscript_label.loc[index_FE, 'Recording_Date']).split("-")
                            label_date = label_date[0] + label_date[1] + label_date[2]
                            label_date = str(label_date).split(" ")
                            label_date = label_date[0]
                            if date == label_date:
                                print("The new spectrogram has a target")
                                for k in range(div):
                                    lab_to_add = unscript_label.loc[index_FE, category]
                                    labels.append(lab_to_add)
                                spectrograms = np.concatenate((spectrograms, spec_t))
                                print(spec.shape, "Spectrogram shape NOISE")
                    
                    print("DATA AUGMENTED")      

                if int(lab_to_add) > 19:
                    y_noise = audio
                    rms = math.sqrt(np.mean(y_noise**2))
                    noise = np.random.normal(0, rms, y_noise.shape[0])
                    y_noise = y_noise + noise                               
                    spec_t= librosa.feature.melspectrogram(
                        y= y_noise, 
                        sr=SAMPLE_RATE, 
                        n_fft=N_FFT, 
                        hop_length=HOP_LENGTH, 
                        n_mels=N_MELS, 
                        fmax = F_MAX,
                        win_length=WIN_LENGTH
                    )
                    spec_t = librosa.power_to_db(spec_t).transpose()
                    #print(spec_t.shape, "Spectrogram shape NOISE")

                    div2 = spec_t.shape[0] // SEQUENCE_LENGTH
                    size_new = int(div2 * SEQUENCE_LENGTH)
                    spec_t = spec_t[:size_new]
                    spec_t = spec_t.reshape((div2, SEQUENCE_LENGTH, N_MELS))
                    
                    print("New Spectrogram augmented")
                    for index_FE, row_FE in unscript_label.iterrows():
                        if id == unscript_label.loc[index_FE, 'participant_ID']:
                            label_date = str(unscript_label.loc[index_FE, 'Recording_Date']).split("-")
                            label_date = label_date[0] + label_date[1] + label_date[2]
                            label_date = str(label_date).split(" ")
                            label_date = label_date[0]
                            if date == label_date:
                                print("The new spectrogram has a target")
                                for k in range(div):
                                    lab_to_add = unscript_label.loc[index_FE, category]
                                    labels.append(lab_to_add)
                                spectrograms = np.concatenate((spectrograms, spec_t))
                                print(spec.shape, "Spectrogram shape NOISE")
                    
                    print("DATA AUGMENTED")    
                
                if (int(lab) == 2) and (control == True):
                    y_noise = audio
                    rms = math.sqrt(np.mean(y_noise**2))
                    noise = np.random.normal(0, rms, y_noise.shape[0])
                    y_noise = y_noise + noise                               
                    spec_t= librosa.feature.melspectrogram(
                        y= y_noise, 
                        sr=SAMPLE_RATE, 
                        n_fft=N_FFT, 
                        hop_length=HOP_LENGTH, 
                        n_mels=N_MELS, 
                        fmax = F_MAX,
                        win_length=WIN_LENGTH
                    )
                    spec_t = librosa.power_to_db(spec_t).transpose()
                    #print(spec_t.shape, "Spectrogram shape NOISE")

                    div2 = spec_t.shape[0] // SEQUENCE_LENGTH
                    size_new = int(div2 * SEQUENCE_LENGTH)
                    spec_t = spec_t[:size_new]
                    spec_t = spec_t.reshape((div2, SEQUENCE_LENGTH, N_MELS))
                    
                    print("New Spectrogram augmented")
                    for index_FE, row_FE in unscript_label.iterrows():
                        if id == unscript_label.loc[index_FE, 'participant_ID']:
                            label_date = str(unscript_label.loc[index_FE, 'Recording_Date']).split("-")
                            label_date = label_date[0] + label_date[1] + label_date[2]
                            label_date = str(label_date).split(" ")
                            label_date = label_date[0]
                            if date == label_date:
                                print("The new spectrogram has a target")
                                for k in range(div):
                                    lab_to_add = unscript_label.loc[index_FE, category]
                                    ind = index_FE
                                    labels.append(lab_to_add)
                                spectrograms = np.concatenate((spectrograms, spec_t))
                                print(spec.shape, "Spectrogram shape NOISE")
                    
                    print("DATA AUGMENTED")

                print("Length Spectrograms: ",spectrograms.shape)
                print("Length labels: ", len(labels))
    
    spectrograms = spectrograms[1:,:,:]
    print(spectrograms.shape, "Spectrograms array")
    print(len(labels))

    labels = np.asarray(labels, dtype=int)
    dataset = TensorDataset(
        torch.from_numpy(spectrograms),
        torch.from_numpy(labels)
    )

    os.chdir("C:\\Users\\ericq\\OneDrive\\Escriptori\\TELECOM\\MSC MATT\\TFM\\The research question\\The Solution\\CNN+RNN")
    dataSaver(dataset, pickleName)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])
    print("Training size:", len(train_set))
    print("Testing size:",len(test_set))

    
    #Print histogram of classes
    n, bins, patches = plt.hist(x=labels,bins='auto',color='#0504aa',alpha=0.7, rwidth=0.85)
    plt.grid(axis='y',alpha=0.75)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Is my data balanced?')
    plt.text(23,45,r'$\mu15, b=3$')
    maxfreq = n.max()
    plt.ylim(ymax=np.ceil(maxfreq/10)*10 if maxfreq % 10 else maxfreq + 10)
    plt.show()

    train_dataloader = torch.utils.data.DataLoader(
        train_set,
        batch_size=128,
        num_workers=2,
        shuffle=True
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_set,
        batch_size=128,
        num_workers=2,
        shuffle=True
    )

    return train_dataloader, test_dataloader, dataset


def dataMLTCreator(train_loader, pickleName, task):
    name =[]
    audio = []
    #spectrograms = np.array()
    labels_dep = []
    labels_anx = []
    
    name = [x[0] for x in os.walk(train_loader)]
    name.pop(0)

    path = train_loader + "\\Spectrograms_RADAR_KCL"
    i = 0
    SAMPLE_RATE = 16000
    N_FFT = 512
    HOP_LENGTH = int(0.01*SAMPLE_RATE)
    N_MELS = 40
    F_MAX = 8000
    WIN_LENGTH= int(0.025*SAMPLE_RATE)
    SEQUENCE_LENGTH= 500

    script_label = labelsMLTLoader("Scripted")
    unscript_label = labelsMLTLoader("Unscripted")
    spectrograms = np.zeros((1,SEQUENCE_LENGTH,N_MELS))
    control = True

    for i, names in enumerate(name):
        #file_path = name[names]
        file_path = names
        files = os.listdir(file_path)
        count = 0

        for j, filename in enumerate(glob.glob(os.path.join(file_path, '*.wav'))):
            #We must read all the folders, but they have really different names.
            x = file_path.split("\\")
            id = x[14]
            y = filename.split("_")
            date = y[3]
            y = date.split("\\")
            date = y[1]
            z = filename.split("-")
            task = z[11]
            if task == "scripted":
                count+=1
                print("Signal processed "+ str(count) + "_" + str(i))
                '''
                if count >= 128:
                    break'''
                audio, sampling_rate = librosa.load(filename, sr=SAMPLE_RATE)
                spec= librosa.feature.melspectrogram(
                    y= audio, 
                    sr=SAMPLE_RATE, 
                    n_fft=N_FFT, 
                    hop_length=HOP_LENGTH, 
                    n_mels=N_MELS, 
                    fmax = F_MAX,
                    win_length=WIN_LENGTH
                )
                spec = librosa.power_to_db(spec).transpose()
                #print(spec.shape, "Spectrogram shape")
                
                div = spec.shape[0] // SEQUENCE_LENGTH
                size_new = int(div * SEQUENCE_LENGTH)
                spec = spec[:size_new]
                spec = spec.reshape((div, SEQUENCE_LENGTH, N_MELS))
            
                print("New Spectrogram")
                for index_FE, row_FE in script_label.iterrows():
                    if id == script_label.loc[index_FE, 'participant_ID']:
                        label_date = str(script_label.loc[index_FE, 'Recording_Date']).split("-")
                        label_date = label_date[0] + label_date[1] + label_date[2]
                        label_date = str(label_date).split(" ")
                        label_date = label_date[0]
                        if date == label_date:
                            print("The new spectrogram has a target")
                            for k in range(div):
                                lab_to_add = script_label.loc[index_FE, "PHQ8"]
                                lab = script_label.loc[index_FE, 'LabelDep']
                                if int(script_label.loc[index_FE, 'LabelDep']) == 2:
                                    if control == True:
                                        control = False
                                    else:
                                        control = True
                                labels_dep.append(lab)
                                labels_anx.append(script_label.loc[index_FE, 'LabelAnx'])
                            spectrograms = np.concatenate((spectrograms, spec))
                            print(spec.shape, "Spectrogram shape")
                '''      
                if int(lab_to_add) > 14:
                    y_noise = audio
                    rms = math.sqrt(np.mean(y_noise**2))
                    noise = np.random.normal(0, rms, y_noise.shape[0])
                    y_noise = y_noise + noise                               
                    spec_t= librosa.feature.melspectrogram(
                        y= y_noise, 
                        sr=SAMPLE_RATE, 
                        n_fft=N_FFT, 
                        hop_length=HOP_LENGTH, 
                        n_mels=N_MELS, 
                        fmax = F_MAX,
                        win_length=WIN_LENGTH
                    )
                    spec_t = librosa.power_to_db(spec_t).transpose()
                    #print(spec_t.shape, "Spectrogram shape NOISE")

                    div2 = spec_t.shape[0] // SEQUENCE_LENGTH
                    size_new = int(div2 * SEQUENCE_LENGTH)
                    spec_t = spec_t[:size_new]
                    spec_t = spec_t.reshape((div2, SEQUENCE_LENGTH, N_MELS))
                    
                    print("New Spectrogram augmented")
                    for index_FE, row_FE in script_label.iterrows():
                        if id == script_label.loc[index_FE, 'participant_ID']:
                            label_date = str(script_label.loc[index_FE, 'Recording_Date']).split("-")
                            label_date = label_date[0] + label_date[1] + label_date[2]
                            label_date = str(label_date).split(" ")
                            label_date = label_date[0]
                            if date == label_date:
                                print("The new spectrogram has a target")
                                for k in range(div):
                                    lab_to_add = script_label.loc[index_FE, 'LabelDep']
                                    labels_dep.append(lab_to_add)
                                    labels_anx.append(script_label.loc[index_FE, 'LabelAnx'])
                                spectrograms = np.concatenate((spectrograms, spec_t))
                                print(spec.shape, "Spectrogram shape NOISE")
                    
                    print("DATA AUGMENTED")      

                if int(lab_to_add) > 19:
                    y_noise = audio
                    rms = math.sqrt(np.mean(y_noise**2))
                    noise = np.random.normal(0, rms, y_noise.shape[0])
                    y_noise = y_noise + noise                               
                    spec_t= librosa.feature.melspectrogram(
                        y= y_noise, 
                        sr=SAMPLE_RATE, 
                        n_fft=N_FFT, 
                        hop_length=HOP_LENGTH, 
                        n_mels=N_MELS, 
                        fmax = F_MAX,
                        win_length=WIN_LENGTH
                    )
                    spec_t = librosa.power_to_db(spec_t).transpose()
                    #print(spec_t.shape, "Spectrogram shape NOISE")

                    div2 = spec_t.shape[0] // SEQUENCE_LENGTH
                    size_new = int(div2 * SEQUENCE_LENGTH)
                    spec_t = spec_t[:size_new]
                    spec_t = spec_t.reshape((div2, SEQUENCE_LENGTH, N_MELS))
                    
                    print("New Spectrogram augmented")
                    for index_FE, row_FE in script_label.iterrows():
                        if id == script_label.loc[index_FE, 'participant_ID']:
                            label_date = str(script_label.loc[index_FE, 'Recording_Date']).split("-")
                            label_date = label_date[0] + label_date[1] + label_date[2]
                            label_date = str(label_date).split(" ")
                            label_date = label_date[0]
                            if date == label_date:
                                print("The new spectrogram has a target")
                                for k in range(div):
                                    lab_to_add = script_label.loc[index_FE, 'LabelDep']
                                    labels_dep.append(lab_to_add)
                                    labels_anx.append(script_label.loc[index_FE, 'LabelAnx'])
                                spectrograms = np.concatenate((spectrograms, spec_t))
                                print(spec.shape, "Spectrogram shape NOISE")
                    
                    print("DATA AUGMENTED")    
                
                if (int(lab) == 2) and (control == True):
                    y_noise = audio
                    rms = math.sqrt(np.mean(y_noise**2))
                    noise = np.random.normal(0, rms, y_noise.shape[0])
                    y_noise = y_noise + noise                               
                    spec_t= librosa.feature.melspectrogram(
                        y= y_noise, 
                        sr=SAMPLE_RATE, 
                        n_fft=N_FFT, 
                        hop_length=HOP_LENGTH, 
                        n_mels=N_MELS, 
                        fmax = F_MAX,
                        win_length=WIN_LENGTH
                    )
                    spec_t = librosa.power_to_db(spec_t).transpose()
                    #print(spec_t.shape, "Spectrogram shape NOISE")

                    div2 = spec_t.shape[0] // SEQUENCE_LENGTH
                    size_new = int(div2 * SEQUENCE_LENGTH)
                    spec_t = spec_t[:size_new]
                    spec_t = spec_t.reshape((div2, SEQUENCE_LENGTH, N_MELS))
                    
                    print("New Spectrogram augmented")
                    for index_FE, row_FE in script_label.iterrows():
                        if id == script_label.loc[index_FE, 'participant_ID']:
                            label_date = str(script_label.loc[index_FE, 'Recording_Date']).split("-")
                            label_date = label_date[0] + label_date[1] + label_date[2]
                            label_date = str(label_date).split(" ")
                            label_date = label_date[0]
                            if date == label_date:
                                print("The new spectrogram has a target")
                                for k in range(div):
                                    lab_to_add = script_label.loc[index_FE, 'LabelDep']
                                    labels_dep.append(lab_to_add)
                                    labels_anx.append(script_label.loc[index_FE, 'LabelAnx'])
                                spectrograms = np.concatenate((spectrograms, spec_t))
                                print(spec.shape, "Spectrogram shape NOISE")
                    
                    print("DATA AUGMENTED")
                '''
                print("Length Spectrograms: ",spectrograms.shape)
                print("Length labels depresion: ", len(labels_dep))
                print("Length labels anxiety: ",len(labels_anx))

            elif task == "unscripted":
                count+=1
                print("Signal processed "+ str(count) + "_" + str(i))
                audio, sampling_rate = librosa.load(filename, sr=SAMPLE_RATE)
                spec= librosa.feature.melspectrogram(
                    y= audio, 
                    sr=SAMPLE_RATE, 
                    n_fft=N_FFT, 
                    hop_length=HOP_LENGTH, 
                    n_mels=N_MELS, 
                    fmax = F_MAX,
                    win_length=WIN_LENGTH
                )
                spec = librosa.power_to_db(spec).transpose()
                #print(spec.shape, "Spectrogram shape")
                
                div = spec.shape[0] // SEQUENCE_LENGTH
                size_new = int(div * SEQUENCE_LENGTH)
                spec = spec[:size_new]
                spec = spec.reshape((div, SEQUENCE_LENGTH, N_MELS))
            
                print("New Spectrogram")
                for index_FE, row_FE in unscript_label.iterrows():
                    if id == unscript_label.loc[index_FE, 'participant_ID']:
                        label_date = str(unscript_label.loc[index_FE, 'Recording_Date']).split("-")
                        label_date = label_date[0] + label_date[1] + label_date[2]
                        label_date = str(label_date).split(" ")
                        label_date = label_date[0]
                        if date == label_date:
                            print("The new spectrogram has a target")
                            for k in range(div):
                                lab_to_add = unscript_label.loc[index_FE, "PHQ8"]
                                lab = unscript_label.loc[index_FE, 'LabelDep']
                                if int(unscript_label.loc[index_FE, 'LabelDep']) == 2:
                                    if control == True:
                                        control = False
                                    else:
                                        control = True
                                labels_dep.append(lab)
                                labels_anx.append(unscript_label.loc[index_FE, 'LabelAnx'])
                            spectrograms = np.concatenate((spectrograms, spec))
                            print(spec.shape, "Spectrogram shape")
                        
                if int(lab_to_add) > 14:
                    y_noise = audio
                    rms = math.sqrt(np.mean(y_noise**2))
                    noise = np.random.normal(0, rms, y_noise.shape[0])
                    y_noise = y_noise + noise                               
                    spec_t= librosa.feature.melspectrogram(
                        y= y_noise, 
                        sr=SAMPLE_RATE, 
                        n_fft=N_FFT, 
                        hop_length=HOP_LENGTH, 
                        n_mels=N_MELS, 
                        fmax = F_MAX,
                        win_length=WIN_LENGTH
                    )
                    spec_t = librosa.power_to_db(spec_t).transpose()
                    #print(spec_t.shape, "Spectrogram shape NOISE")

                    div2 = spec_t.shape[0] // SEQUENCE_LENGTH
                    size_new = int(div2 * SEQUENCE_LENGTH)
                    spec_t = spec_t[:size_new]
                    spec_t = spec_t.reshape((div2, SEQUENCE_LENGTH, N_MELS))
                    
                    print("New Spectrogram augmented")
                    for index_FE, row_FE in unscript_label.iterrows():
                        if id == unscript_label.loc[index_FE, 'participant_ID']:
                            label_date = str(unscript_label.loc[index_FE, 'Recording_Date']).split("-")
                            label_date = label_date[0] + label_date[1] + label_date[2]
                            label_date = str(label_date).split(" ")
                            label_date = label_date[0]
                            if date == label_date:
                                print("The new spectrogram has a target")
                                for k in range(div):
                                    lab_to_add = unscript_label.loc[index_FE, 'LabelDep']
                                    labels_dep.append(lab_to_add)
                                    labels_anx.append(unscript_label.loc[index_FE, 'LabelAnx'])
                                spectrograms = np.concatenate((spectrograms, spec_t))
                                print(spec.shape, "Spectrogram shape NOISE")
                    
                    print("DATA AUGMENTED")      

                if int(lab_to_add) > 19:
                    y_noise = audio
                    rms = math.sqrt(np.mean(y_noise**2))
                    noise = np.random.normal(0, rms, y_noise.shape[0])
                    y_noise = y_noise + noise                               
                    spec_t= librosa.feature.melspectrogram(
                        y= y_noise, 
                        sr=SAMPLE_RATE, 
                        n_fft=N_FFT, 
                        hop_length=HOP_LENGTH, 
                        n_mels=N_MELS, 
                        fmax = F_MAX,
                        win_length=WIN_LENGTH
                    )
                    spec_t = librosa.power_to_db(spec_t).transpose()
                    #print(spec_t.shape, "Spectrogram shape NOISE")

                    div2 = spec_t.shape[0] // SEQUENCE_LENGTH
                    size_new = int(div2 * SEQUENCE_LENGTH)
                    spec_t = spec_t[:size_new]
                    spec_t = spec_t.reshape((div2, SEQUENCE_LENGTH, N_MELS))
                    
                    print("New Spectrogram augmented")
                    for index_FE, row_FE in unscript_label.iterrows():
                        if id == unscript_label.loc[index_FE, 'participant_ID']:
                            label_date = str(unscript_label.loc[index_FE, 'Recording_Date']).split("-")
                            label_date = label_date[0] + label_date[1] + label_date[2]
                            label_date = str(label_date).split(" ")
                            label_date = label_date[0]
                            if date == label_date:
                                print("The new spectrogram has a target")
                                for k in range(div):
                                    lab_to_add = unscript_label.loc[index_FE, 'LabelDep']
                                    labels_dep.append(lab_to_add)
                                    labels_anx.append(unscript_label.loc[index_FE, 'LabelAnx'])
                                spectrograms = np.concatenate((spectrograms, spec_t))
                                print(spec.shape, "Spectrogram shape NOISE")
                    
                    print("DATA AUGMENTED")    
                
                if (int(lab) == 2) and (control == True):
                    y_noise = audio
                    rms = math.sqrt(np.mean(y_noise**2))
                    noise = np.random.normal(0, rms, y_noise.shape[0])
                    y_noise = y_noise + noise                               
                    spec_t= librosa.feature.melspectrogram(
                        y= y_noise, 
                        sr=SAMPLE_RATE, 
                        n_fft=N_FFT, 
                        hop_length=HOP_LENGTH, 
                        n_mels=N_MELS, 
                        fmax = F_MAX,
                        win_length=WIN_LENGTH
                    )
                    spec_t = librosa.power_to_db(spec_t).transpose()
                    #print(spec_t.shape, "Spectrogram shape NOISE")

                    div2 = spec_t.shape[0] // SEQUENCE_LENGTH
                    size_new = int(div2 * SEQUENCE_LENGTH)
                    spec_t = spec_t[:size_new]
                    spec_t = spec_t.reshape((div2, SEQUENCE_LENGTH, N_MELS))
                    
                    print("New Spectrogram augmented")
                    for index_FE, row_FE in unscript_label.iterrows():
                        if id == unscript_label.loc[index_FE, 'participant_ID']:
                            label_date = str(unscript_label.loc[index_FE, 'Recording_Date']).split("-")
                            label_date = label_date[0] + label_date[1] + label_date[2]
                            label_date = str(label_date).split(" ")
                            label_date = label_date[0]
                            if date == label_date:
                                print("The new spectrogram has a target")
                                for k in range(div):
                                    lab_to_add = unscript_label.loc[index_FE, 'LabelDep']
                                    labels_dep.append(lab_to_add)
                                    labels_anx.append(unscript_label.loc[index_FE, 'LabelAnx'])
                                spectrograms = np.concatenate((spectrograms, spec_t))
                                print(spec.shape, "Spectrogram shape NOISE")
                    
                    print("DATA AUGMENTED")

                print("Length Spectrograms: ",spectrograms.shape)
                print("Length labels depresion: ", len(labels_dep))
                print("Length labels anxiety: ",len(labels_anx))


    spectrograms = spectrograms[1:,:,:]
    print(spectrograms.shape, "Spectrograms array")
    print(len(labels_anx))
    print(len(labels_dep))

    labels_anx = np.asarray(labels_anx, dtype=int)
    labels_dep = np.asarray(labels_dep, dtype=int)

    dataset = TensorDataset(
        torch.from_numpy(spectrograms),
        torch.from_numpy(labels_dep),
        torch.from_numpy(labels_anx)
    )

    os.chdir("C:\\Users\\ericq\\OneDrive\\Escriptori\\TELECOM\\MSC MATT\\TFM\\The research question\\The Solution\\CNN+RNN")
    dataSaver(dataset, pickleName)

    return dataset, spectrograms, labels_dep, labels_anx
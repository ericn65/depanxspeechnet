'''
SPECTOGRAMS LOADER
No transformations done, everything is just to test if we have the decent and necessary data.
'''

import os
from data_Loader import dataLoader, datatoSpec

train_loader = "C:\\Users\\ericq\\OneDrive\\Escriptori\\TELECOM\\MSC MATT\\TFM\\The research question\\The Solution\\CNN+RNN\\Raw_Data_v2\\RADAR-MDD-KCL-s1\\RADAR-MDD-KCL-s1"
test_loader = ""

name =[]
data = []
audio = []
spectrograms = []
labels = []
input_lengths = []
label_lengths = []

name = [x[0] for x in os.walk(train_loader)]

name.pop(0)

for names in range(len(name)):
    print(str(name[names]))

for names in range(len(name)):
#    file_path = train_loader + "\\" + str(name[names]) + "/questionnaire_audio"
    file_path = name[names]
    audio = dataLoader(file_path)
    data.append(audio)

backup = data
for i in range(len(data)-1, -1, -1):
    if not data[i]:
        del data[i]

path = train_loader + "\\Spectrograms_RADAR_KCL"
try:
    os.chdir(path)
except:
    os.mkdir(path)
    os.chdir(path)

spectrograms = datatoSpec(data, 16000, path)
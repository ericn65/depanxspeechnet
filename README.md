# DepAnxSpeechNet
Joint Learning of Depression and Anxiety direct from speech signals <br>

## Abstract
Advances in digital health and phenotyping technologies are crucial to ensureincreased access to high-quality mental health support services and treatment. Speech is 
uniquely placed in this regard, as no other mobile health signal contains its singular combination of cognitive, neuromuscular and physiological information. It is 
this complexity which makes speech a suitable marker for different mentalhealth conditions. However, much research exploring  links  between  speech  and  depression  
is  limited,  and  co-morbidities  with conditions  such  as  anxiety  have  not  been  exploited  to  help  improve machine  learning models. The purpose of this 
project is to jointly learn depression and anxiety directly from speech signals. <br> 
For  this  project,  speech  signals  were  split  into  segments  that  were  converted  into  Mel-spectrograms. Automatic feature extraction was performed using a 
CNN-LSTM model that can  classify into5 severities  of depression. With  transfer  learning,  this  model was then usedas  a pre-trained  model  for other tasks,  such 
as  classifying  speech  signals  into different 4 severities  of  anxiety  or  improving  models for  both  co-morbiditiesin  different languages.  Finally,  a  
Multi-Task  learning  model  is  used  to  jointly  detect  depression  and anxiety. <br>
Models that use transfer learning to detectanxiety achieve an improvement from 67% to 72% of accuracy, while multi-Task learning models achieve an accuracy of 71% for 
both co-morbidities, anxiety and depression. <br>
The experiments show promising results, discussing the viability of jointly detecting mental health conditions such as depression and anxiety as well as exploiting the 
viability of using models pre-trained for just one condition, language or task to fine-tune a model for another condition, language or task, demonstrating that co-
morbidities can help to improve models for joint learning severities directly from speech signals. <br><br>
## How to use
Libraries contain the code necessary to upload and save datasets, creating them to Mel-Spectrograms (<font color=blue>datasetcreator.py</font<); to implement the 
models used for this project (<font color=blue>networks_v2.py</font>); and to optimize, test and train the model (<font color=blue>optimizer.py</font>).<br>
ipynb archives are self-described to create and test models for Transfer Learning, Multi-task learning and test the original models over PHQ-8 and GAD-7. <br>
Models include archives able to pre-train and fine-tune using as any other nn.Module. They are pre-train over RADAR-MDD data, not available in this repository. 

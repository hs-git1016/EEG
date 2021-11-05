from numpy.core.fromnumeric import shape
from tensorflow.keras.models import load_model
import os 
from mne.io import read_raw_edf
from mne import read_annotations
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D,LSTM,MaxPooling1D,AveragePooling1D,SpatialDropout1D,Dropout,Dense
from tensorflow.keras.losses import categorical_crossentropy
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

def hyp_reader(data):
    desc=data.description
    dur=data.duration
    dur=dur.astype(int)
    def rep(index):
        temp = np.array([desc[index]]*(int(dur[index]/30)))
        return temp
    hyp = rep(0)
    for i in range(1,len(dur)-1):
        hyp=np.concatenate((hyp,rep(i)),axis=0)
    return hyp
    

def read_EDF(psg,hyp):
    data  = read_raw_edf(os.getcwd() + '/Data/'+psg+'.edf')
    raw_data = data.get_data()
    data_hyp  = read_annotations(os.getcwd() + '/Data/'+hyp+'.edf')
    l1 = raw_data[0]
    l2 = raw_data[1]
    l3 = hyp_reader(data_hyp)
    return l1,l2,l3

def merge_data(fpz,pz,label,psg,hyp):
    temp1,temp2,temp3=read_EDF(psg,hyp)
    fpz=np.concatenate((fpz,temp1),axis=0)
    pz=np.concatenate((pz,temp2),axis=0)
    label=np.concatenate((label,temp3),axis=0)
    return fpz,pz,label

def data_reshape(fpz,pz,label):
    fpz= np.reshape(fpz,(label.shape[0],samples,1))
    pz= np.reshape(pz,(label.shape[0],samples,1))
    return fpz,pz


fs = 100
epochs = 30
samples = fs*epochs
fpz,pz,label = read_EDF('SC4002E0-PSG','SC4002EC-Hypnogram')

fpz,pz = data_reshape(fpz,pz,label)

print(type(fpz))
print(fpz.shape[2])

encoder = LabelEncoder()
encoder.fit(label)
label = encoder.transform(label)
label = to_categorical(label)


model = load_model( os.getcwd() + '/model/test_model.h5')
pred_fpz = model.predict(fpz[0:2])

y_test_label = np.argmax(pred_fpz,axis=-1)
print(y_test_label)


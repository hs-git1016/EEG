# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 11:13:47 2020

@author: Aadarsh Srivastava
"""
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
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def data_reshape(fpz,label):
    fpz= np.reshape(fpz,(label.shape[0],samples,1))
    return fpz



def model_single():
    model = Sequential()
    
    model.add(Conv1D(filters=128,kernel_size=7,activation='relu',input_shape=(36,1)))
    model.add(MaxPooling1D(pool_size=3,strides=2))
    model.add(BatchNormalization())
    model.add(Conv1D(filters=192,kernel_size=5,activation='relu',input_shape=(128,1)))
    model.add(Conv1D(filters=256,kernel_size=5,activation='relu',input_shape=(192,1)))
    # model.add(MaxPooling1D(pool_size=3,strides=2))
    # model.add(Conv1D(filters=256,kernel_size=3,activation='relu',input_shape=(384,1)))
    # model.add(Conv1D(filters=256,kernel_size=3,activation='relu',input_shape=(256,1)))
    model.add(MaxPooling1D(pool_size=3,strides=2))
    
    # model.add(SpatialDropout1D(0.5))
    model.add(Flatten())
    # model.add(LSTM(256,dropout=0.3,recurrent_dropout=0.3))
    # model.add(Dense(512,activation='relu'))
    # model.add(Dropout(0.8))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.8))
    model.add(Dense(5, activation='softmax'))

    model.compile(optimizer='adam', loss=categorical_crossentropy,metrics=['accuracy'])
    return model

PATH_TRAIN_FILE = "C:/Users/DELL/Desktop/EEG/children"
data = np.load(os.path.join(PATH_TRAIN_FILE, '03.npz'))
x = data['x']
label = data['y']
fs = int(data['fs'])

epochs = 30
samples = 36
# x = norm(x)

fpz = data_reshape(x,label)
print(fpz.shape)
print(label.shape)

print(label[2])
encoder = LabelEncoder()
encoder.fit(label)
label = encoder.transform(label)
print(label.shape)
print(label[2])
label = to_categorical(label)
print(label.shape)


model = load_model( os.getcwd() + '/model/test_model.h5')
pred_fpz = model.predict(fpz)

label_fpz = np.argmax(pred_fpz,axis=-1)
y_label = np.argmax(label,axis=-1)
print(label_fpz)
print(y_label)
print('Accuracy : {} %'.format(accuracy_score(y_label,label_fpz)*100))

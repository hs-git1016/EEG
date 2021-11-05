# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 11:13:47 2020

@author: Aadarsh Srivastava
"""

from mne.io import read_raw_edf
from mne import read_annotations
import numpy as np

import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers, losses, activations, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D,LSTM,MaxPooling1D,AveragePooling1D,SpatialDropout1D,Dropout,Dense,Flatten,BatchNormalization,GRU
from tensorflow.keras.losses import categorical_crossentropy
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from glob import glob

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def data_reshape(fpz,label):
    fpz= np.reshape(fpz,(label.shape[0],samples,1))
    return fpz


def model_single():
    model = Sequential()
    
    model.add(Conv1D(filters = 128,kernel_size = 7,activation = 'relu',input_shape=(36,1)))
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
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# conv1d (Conv1D)              (None, 30, 128)           1024
# _________________________________________________________________
# max_pooling1d (MaxPooling1D) (None, 14, 128)           0
# _________________________________________________________________
# batch_normalization (BatchNo (None, 14, 128)           512
# _________________________________________________________________
# conv1d_1 (Conv1D)            (None, 10, 192)           123072
# _________________________________________________________________
# conv1d_2 (Conv1D)            (None, 6, 256)            246016
# _________________________________________________________________
# max_pooling1d_1 (MaxPooling1 (None, 2, 256)            0
# _________________________________________________________________
# flatten (Flatten)            (None, 512)               0
# _________________________________________________________________
# dense (Dense)                (None, 128)               65664
# _________________________________________________________________
# dropout (Dropout)            (None, 128)               0
# _________________________________________________________________
# dense_1 (Dense)              (None, 5)                 645



PATH_TRAIN_FILE = "C:/Users/DELL/Desktop/EEG/children"
psg_files = glob(os.path.join(PATH_TRAIN_FILE,'*.npz'))


def load_data(file):
    data = np.load(file)
    x_data = data['x']
    label_data = data['y']
    return x_data,label_data


x_1,label_1 = load_data(psg_files[0])
x_2,label_2 = load_data(psg_files[1])
# x_3,label_3 = load_data(psg_files[2])

x = np.concatenate([x_1,x_2])
label = np.concatenate([label_1,label_2])


print(x.shape)
epochs = 30
samples = 36
class_dict={0:"W",1:"N1",2:"N2",3:"N3",4:"REM",5:"UNKNOWN"}
ann2label={"Sleep stage W":0,"Sleep stage 1":1,"Sleep stage 2":2,"Sleep stage 3":3,"Sleep stage 4":3,"Sleep stage R":5}


# x = norm(x)

fpz = data_reshape(x,label)
# pz = data_reshape(pz,label)
print(fpz.shape)
print(label.shape)

print(label[1134])
encoder = LabelEncoder()
encoder.fit(label)
label = encoder.transform(label)
print(label.shape)
print(label[1134])
label = to_categorical(label)
print(label.shape)

x_train,x_test,y_train,y_test = train_test_split(fpz,label,test_size=0.2,random_state=1)
x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,test_size=0.25,random_state=1)

print('train:',x_train.shape,y_train.shape)
print('val:',x_val.shape,y_val.shape)
print('test:',x_test.shape,y_test.shape)

model_fpz = model_single()
model_fpz.fit(x=x_train,y=y_train,epochs=20,verbose=1)#,callbacks=[EarlyStopping(patience=3,restore_best_weights=True)])
save_path = os.getcwd() +'/model/test_model.h5'
model_fpz.save(save_path)
pred_fpz = model_fpz.predict(x_test,verbose=1)
label_fpz = np.argmax(pred_fpz,axis=-1)
y_test_label = np.argmax(y_test,axis=-1)
print('Accuracy : {} %'.format(accuracy_score(y_test_label,label_fpz)*100))

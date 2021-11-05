# from glob import glob
# import os
# import numpy as np
# PATH_TRAIN_FILE = "C:/Users/DELL/Desktop/EEG/children"
# psg_files = glob(os.path.join(PATH_TRAIN_FILE,'*.npz'))

# def load_data(file):
#     data = np.load(file)
#     x_data = data['x']
#     label_data = data['y']
#     return x_data,label_data

# x_1,label_1 = load_data(psg_files[0])

# # def maxminnorm(array):
# #     maxcols=array.max(axis=0)
# #     mincols=array.min(axis=0)
# #     data_shape = array.shape
# #     data_rows = data_shape[0]
# #     data_cols = data_shape[1]
# #     t=np.empty((data_rows,data_cols))
# #     for i in range(data_cols):
# #         t[:,i]=(array[:,i]-mincols[i])/(maxcols[i]-mincols[i])
# #     return t

# samples = 6
# print(x_1.shape)
# print(x_1.shape[0])
# def data_reshape(fpz,label):
#     fpz= np.reshape(fpz,(label.shape[0],samples,6))
#     return fpz


# x_1 = data_reshape(x_1,label_1)
# print(x_1.shape)

# print(x_1)


from tensorflow.keras.models import load_model
import os 
from mne.io import read_raw_edf
from mne import read_annotations
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D,LSTM,MaxPooling1D,AveragePooling1D,SpatialDropout1D,Dropout,Dense
from tensorflow.keras.losses import categorical_crossentropy
model = load_model( os.getcwd() + '/model/test_model.h5')
model.summary()
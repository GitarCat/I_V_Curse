from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
np.random.seed(7)

import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard, Callback
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten,concatenate,GlobalMaxPooling1D
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.layers import LSTM, Activation, GRU, RNN
#from keras.layers.core import Flatten
from keras.layers.convolutional import Conv1D, Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import MaxPooling1D, MaxPooling2D
from keras import regularizers
from keras import initializers
from keras import Input
from keras import Model
import scipy.io as sio
from IPython.display import clear_output
import random
from sklearn import manifold

#normalization
def NormalizationV(x):
    x = np.array(x, dtype='float32')
    for i in range(0, x.shape[0]):
        for j in range(x.shape[1]):
            x[i,j]=x[i,j]/(13*37.78)
    return x

def NormalizationI(x):
    x=np.array(x,dtype='float32')
    for i in range(0, x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] = x[i, j] / 9.12
    return x


#read data
V_data=sio.loadmat('V_train.mat')
V_data=V_data['V_train']
V_data=np.array(V_data)
I_data=sio.loadmat('I_train.mat')
I_data=I_data['I_train']
I_data=np.array(I_data)
IrrT_data=sio.loadmat('IrrT_train.mat')
IrrT_data=IrrT_data['IrrT_train']
IrrT_data=np.array(IrrT_data)
label=sio.loadmat('label_train.mat')
label=label['label_train']


#shuffle
num=len(label)
index = [i for i in range(num)]
random.shuffle(index)
index = np.array(index)

#label data process
label=label.reshape(-1)
label=np.array(label)
label=np_utils.to_categorical(label, num_classes=13)
label=label[index]


#Irr&T data process
IrrT_data=IrrT_data.reshape((IrrT_data.shape[0],IrrT_data.shape[1]))
IrrT_data=IrrT_data[index]


#V&I dataprocess
V_data=NormalizationV(V_data)
I_data=NormalizationI(I_data)
#V_data=seg(V_data)
#I_data=seg(I_data)
V_data=V_data.reshape((V_data.shape[0],V_data.shape[1],1))
I_data=I_data.reshape((I_data.shape[0],I_data.shape[1],1))
V_data=V_data[index]
I_data=I_data[index]

#model
input_V=Input(shape=(1000,1),name='input_V')
input_I=Input(shape=(1000,1),name='input_I')
x=concatenate([input_V,input_I])
x=Conv1D(filters=128, kernel_size=8, activation='relu',
                 kernel_initializer=initializers.he_normal())(x)
x=MaxPooling1D(pool_size=2, strides=2)(x)
x=BatchNormalization()(x)
x=Dropout(0.2)(x)

x=GRU(256, return_sequences=False, recurrent_dropout=0.2)(x)
x=BatchNormalization()(x)
x=Dropout(0.2)(x)

input_IrrT=Input(shape=(2,),name='input_IrrT')
x1=concatenate([x,input_IrrT],axis=1)
x1=Dense(128, activation='relu')(x1)
x1=Dense(64, activation='relu')(x1)

output_m=Dense(13, kernel_regularizer=regularizers.l2(0.0015),activation='softmax',name='output_m')(x1)
model=Model(inputs=[input_V,input_I,input_IrrT],outputs=[output_m])
opt = Adam(lr=1e-4)
model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])

#train
model.fit({'input_V':V_data,'input_I':I_data,'input_IrrT':IrrT_data}, {'output_m':label}, epochs=200, batch_size=64, validation_split=0.2)

model.save('2yue26_model.h5')
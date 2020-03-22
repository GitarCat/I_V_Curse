from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd

np.random.seed(7)
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard, Callback
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, concatenate, GlobalMaxPooling1D, Reshape
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.layers import LSTM, Activation, GRU, RNN
# from keras.layers.core import Flatten
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


# normalization
def NormalizationV(x):
    x = np.array(x, dtype='float32')
    return x / (13 * 37.78)


def NormalizationI(x):
    x = np.array(x, dtype='float32')
    return x / 9.12


def main():
    # read data
    V_data = sio.loadmat('V_train.mat')
    V_data = V_data['V_train']
    V_data = np.array(V_data)
    I_data = sio.loadmat('I_train.mat')
    I_data = I_data['I_train']
    I_data = np.array(I_data)
    IrrT_data = sio.loadmat('IrrT_train.mat')
    IrrT_data = IrrT_data['IrrT_train']
    IrrT_data = np.array(IrrT_data)
    label = sio.loadmat('label_train.mat')
    label = label['label_train']


    #shuffle
    index = np.arange(0, len(label))

    random.shuffle(index)

    #label data process
    label = label.reshape(-1)
    label = np_utils.to_categorical(label, num_classes=13)
    label = label[index]

    #Irr&T data process
    Ir_data = np.zeros((len(label), 1000))
    T_data = np.zeros((len(label), 1000))
    for i in range(1000):
        Ir_data[:, i] = IrrT_data[:, 0] / 775.29
        T_data[:, i] = IrrT_data[:, 1] / 51.98
    Ir_data = Ir_data.reshape((Ir_data.shape[0], Ir_data.shape[1], 1))
    T_data = T_data.reshape((T_data.shape[0], T_data.shape[1], 1))
    Ir_data = Ir_data[index]
    T_data = T_data[index]

    #V&I dataprocess
    V_data = NormalizationV(V_data)
    I_data = NormalizationI(I_data)
    V_data = V_data.reshape((V_data.shape[0], V_data.shape[1], 1))
    I_data = I_data.reshape((I_data.shape[0], I_data.shape[1], 1))
    V_data = V_data[index]
    I_data = I_data[index]



    #model
    input_V = Input(shape=(1000, 1), name='input_V')
    input_I = Input(shape=(1000, 1), name='input_I')
    input_Ir = Input(shape=(1000, 1), name='input_Ir')
    input_T = Input(shape=(1000, 1), name='input_T')

    Conv_Model = concatenate([input_V, input_I, input_Ir, input_T], axis=1)


    Conv_Model = Conv1D(filters=64, kernel_size=4, activation='relu',
               kernel_initializer=initializers.he_normal())(Conv_Model)


    Conv_Model = MaxPooling1D(pool_size=2, strides=2)(Conv_Model)
    Conv_Model = BatchNormalization()(Conv_Model)
    Conv_Model = Dropout(0.2)(Conv_Model)

    Conv_Model = Conv1D(filters=128, kernel_size=8, activation='relu',
                kernel_initializer=initializers.he_normal())(Conv_Model)
    Conv_Model = MaxPooling1D(pool_size=2, strides=2)(Conv_Model)
    Conv_Model = BatchNormalization()(Conv_Model)
    Conv_Model = Dropout(0.2)(Conv_Model)

    # Conv_Model = Flatten()(Conv_Model)
    print(Conv_Model.shape)
    GRU_Model = GRU(256, return_sequences=True, recurrent_dropout=0.2)(Conv_Model)
    print(GRU_Model.shape)
    GRU_Model = GRU(128, return_sequences=True, recurrent_dropout=0.2)(Conv_Model)
    print(GRU_Model.shape)
    exit()
    # GRU_Model = BatchNormalization()(GRU_Model)
    # GRU_Model = Dropout(0.2)(GRU_Model)

    #
    # input_IrrT = Input(shape=(2,), name='input_IrrT')
    # Fc_Model = concatenate([GRU_Model, input_IrrT], axis=1)

    # Fc_Model = Dense(128, activation='relu')(Conv_Model)
    # Fc_Model = Dense(64, activation='relu')(Fc_Model)
    # Fc_Model = Dense(32, activation='relu')(Fc_Model)

    # Attention_Model = Attention()(GRU_Model)


    output_m = Dense(13, kernel_regularizer=regularizers.l2(0.0015), activation='softmax', name='output_m')(GRU_Model)
    model = Model(inputs=[input_V, input_I, input_Ir, input_T], outputs=[output_m])
    opt = Adam(lr=1e-4)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    # train
    model.fit({'input_V': V_data, 'input_I': I_data, 'input_Ir': Ir_data, 'input_T': T_data}, {'output_m': label}, epochs=200,
              batch_size=16, validation_split=0.2)

    model.save('test.h5')

if __name__ == '__main__':
    main()
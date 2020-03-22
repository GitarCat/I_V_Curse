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
    IrrT_data = IrrT_data.reshape((IrrT_data.shape[0], IrrT_data.shape[1]))
    IrrT_data[:, 0] = IrrT_data[:, 0] / 775.29
    IrrT_data[:, 1] = IrrT_data[:, 1] / 51.98
    IrrT_data = IrrT_data[index]

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
    input_IrrT = Input(shape=(2,), name='input_IrrT')

    Conv_Model = concatenate([input_V, input_I], axis=1)

    Conv_Model = Conv1D(filters=32, kernel_size=2, activation='relu',
               kernel_initializer=initializers.he_normal())(Conv_Model)
    Conv_Model = MaxPooling1D(pool_size=2, strides=2)(Conv_Model)
    Conv_Model = Conv1D(filters=32, kernel_size=8, activation='relu',
                        kernel_initializer=initializers.he_normal())(Conv_Model)
    Conv_Model = MaxPooling1D(pool_size=2, strides=2)(Conv_Model)
    Conv_Model = Conv1D(filters=32, kernel_size=8, activation='relu',
                        kernel_initializer=initializers.he_normal())(Conv_Model)
    Conv_Model = MaxPooling1D(pool_size=2, strides=2)(Conv_Model)

    Conv_Model = BatchNormalization()(Conv_Model)
    Conv_Model = Dropout(0.2)(Conv_Model)

    Conv_shape = int(Conv_Model.shape[1]) * int(Conv_Model.shape[2]) + 2
    Conv_Model = Flatten()(Conv_Model)
    Conv_Model = concatenate([Conv_Model, input_IrrT], axis=1)
    Conv_Model = Reshape((Conv_shape, 1))(Conv_Model)

    GRU_Model = GRU(32, return_sequences=True, recurrent_dropout=0.2)(Conv_Model)
    GRU_Model = BatchNormalization()(GRU_Model)
    GRU_Model = Dropout(0.2)(GRU_Model)
    GRU_Model = Flatten()(GRU_Model)

    Fc_Model = Dense(64, activation='relu')(GRU_Model)
    Fc_Model = Dense(32, activation='relu')(Fc_Model)

    output_m = Dense(13, kernel_regularizer=regularizers.l2(0.0015), activation='softmax', name='output_m')(Fc_Model)
    model = Model(inputs=[input_V, input_I, input_IrrT], outputs=[output_m])
    opt = Adam(lr=1e-4)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    # train
    model.fit({'input_V': V_data, 'input_I': I_data, 'input_IrrT': IrrT_data}, {'output_m': label}, epochs=200,
              batch_size=16, validation_split=0.2)

    model.save('CNN_GRU.h5')

if __name__ == '__main__':
    main()
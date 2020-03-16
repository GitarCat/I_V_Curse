import numpy as np
import pandas as pd
import keras
import random


from keras import Input
from keras import Model
from keras import regularizers
from keras import initializers
from keras import optimizers
from keras.layers import Dense, Conv1D, Dropout, concatenate, Flatten, LSTM
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import MaxPooling1D

import scipy.io as sio
np.random.seed(5)

def main():
    I_Dataset = pd.read_csv('./dataset/I_data.csv').values
    V_Dataset = pd.read_csv('./dataset/V_Data.csv').values
    Ir_T_Data = pd.read_csv('./dataset/Ir_T.csv').values
    Label = pd.read_csv('./dataset/label.csv').transpose()
    index = np.arange(0, Label.shape[1])
    Data_frac = int(Label.shape[1]*0.8)
    Label = Label.values[0].reshape(-1) - 1

    random.shuffle(index)

    I_Dataset = np.array(I_Dataset, dtype='float32') / 9.12
    V_Dataset = np.array(V_Dataset, dtype='float32') / (13*37.78)
    I_Dataset = np.expand_dims(I_Dataset, axis=2)
    V_Dataset = np.expand_dims(V_Dataset, axis=2)

    I_Dataset = I_Dataset[index]
    V_Dataset = V_Dataset[index]
    Ir_T_Data = Ir_T_Data[index]
    Label_Onehot = np_utils.to_categorical(Label, num_classes=5)[index]

    V_train = Input(shape=(1000, 1), name='V_train')
    I_train = Input(shape=(1000, 1), name='I_train')
    Ir_T_train = Input(shape=(2,), name='Ir_T_train')
    model_input = [V_train, I_train, Ir_T_train]

    IV_train = concatenate([V_train, I_train])

    Conv_Model = Conv1D(filters=64, kernel_size=10, activation='relu',
                 kernel_initializer=initializers.he_normal())(IV_train)

    Conv_Model = MaxPooling1D(pool_size=2, strides=2)(Conv_Model)

    Conv_Model = Conv1D(filters=128, kernel_size=8, activation='relu',
                        kernel_initializer=initializers.he_normal())(Conv_Model)

    Conv_Model = MaxPooling1D(pool_size=2, strides=2)(Conv_Model)

    Conv_Model = Conv1D(filters=256, kernel_size=8, activation='relu',
                        kernel_initializer=initializers.he_normal())(Conv_Model)

    Conv_Model = MaxPooling1D(pool_size=2, strides=2)(Conv_Model)

    Conv_Model = Conv1D(filters=512, kernel_size=4, activation='relu',
                        kernel_initializer=initializers.he_normal())(Conv_Model)

    Conv_Model = MaxPooling1D(pool_size=2, strides=2)(Conv_Model)

    Conv_Model = BatchNormalization()(Conv_Model)
    Conv_Model = Dropout(0.65)(Conv_Model)

    Conv_Model = Flatten()(Conv_Model)


    Fc_Model = concatenate([Conv_Model, Ir_T_train], axis=1)

    Fc_Model = Dense(128, activation='relu')(Fc_Model)
    Fc_Model = Dense(64, activation='relu')(Fc_Model)
    Fc_Out = Dense(5, kernel_regularizer=regularizers.l2(0.0015), activation='softmax', name='Fc_Out')(Fc_Model)

    model = Model(inputs=model_input, outputs=[Fc_Out])
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.Adam(lr=1e-4),
                  metrics=['accuracy'])

    model.fit({'V_train': V_Dataset, 'I_train': I_Dataset, 'Ir_T_train': Ir_T_Data}, {'Fc_Out': Label_Onehot},
              epochs=400,
              batch_size=32,
              validation_split=0.2)

    model.save('model1.h5')





if __name__ == '__main__':
    main()

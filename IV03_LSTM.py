import numpy as np
import pandas as pd
import keras
import random
import tensorflow as tf
import os

from keras import Input
from keras import Model
from keras import regularizers
from keras import initializers
from keras import optimizers
from keras.layers import Dense, Conv1D, Dropout, concatenate, Flatten, LSTM, Reshape
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import MaxPooling1D


os.environ["CUDA_VISIBLE_DEVICES"]="-1"

def main():

    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    I_Dataset = pd.read_csv('./dataset/I_data.csv').values
    V_Dataset = pd.read_csv('./dataset/V_Data.csv').values
    Ir_T_Data = pd.read_csv('./dataset/Ir_T.csv').values
    Label = pd.read_csv('./dataset/label.csv').transpose()
    index = np.arange(0, Label.shape[1])
    Label = Label.values[0].reshape(-1) - 1

    random.shuffle(index)

    Ir_T_Data[:, 0] = Ir_T_Data[:, 0] / 775.75
    Ir_T_Data[:, 1] = Ir_T_Data[:, 1] / 52.734
    I_Dataset = np.array(I_Dataset, dtype='float32') / 9.12
    V_Dataset = np.array(V_Dataset, dtype='float32') / (13*37.78)

    I_Dataset = np.expand_dims(I_Dataset, axis=2)
    V_Dataset = np.expand_dims(V_Dataset, axis=2)

    Ir_T_Data = np.expand_dims(Ir_T_Data, axis=2)
    print(Ir_T_Data.shape)

    I_Dataset = I_Dataset[index]
    V_Dataset = V_Dataset[index]
    Ir_T_Data = Ir_T_Data[index]
    Label_Onehot = np_utils.to_categorical(Label, num_classes=5)[index]

    V_train = Input(shape=(1000, 1), name='V_train')
    I_train = Input(shape=(1000, 1), name='I_train')
    Ir_T_train = Input(shape=(2, 1), name='Ir_T_train')
    model_input = [V_train, I_train, Ir_T_train]


    IV_train = concatenate([V_train, I_train], axis=1)

    LSTM_Model = concatenate([IV_train, Ir_T_train],axis=1)

    # Conv_Model = Conv1D(filters=64, kernel_size=10, activation='relu',
    #                     kernel_initializer=initializers.he_normal())(IV_train)
    #
    # Conv_Model = MaxPooling1D(pool_size=2, strides=2)(Conv_Model)
    #
    # Conv_Model = Conv1D(filters=128, kernel_size=10, activation='relu',
    #                     kernel_initializer=initializers.he_normal())(Conv_Model)
    #
    # Conv_Model = MaxPooling1D(pool_size=2, strides=2)(Conv_Model)
    #
    # Conv_Model = BatchNormalization()(Conv_Model)
    # Conv_Model = Dropout(0.6)(Conv_Model)


    LSTM_Model = LSTM(256, activation='relu')(LSTM_Model)
    LSTM_Model = BatchNormalization()(LSTM_Model)
    LSTM_Model = Dropout(0.2)(LSTM_Model)




    # Conv_shape = Conv_Model.shape[1] * Conv_Model.shape[2] + 2
    #
    # Conv_Model = Flatten()(Conv_Model)
    #
    # Conv_Model = concatenate([Conv_Model, Ir_T_train], axis=1)
    #
    # Conv_Model = Reshape((Conv_shape, 1))(Conv_Model)
    # LSTM_Model = LSTM(128, activation='relu')(Conv_Model)
    # LSTM_Model = Dropout(0.5)(LSTM_Model)


    Fc_Model = Dense(128, activation='relu')(LSTM_Model)
    Fc_Model = Dense(64, activation='relu')(Fc_Model)
    Fc_Out = Dense(5, kernel_regularizer=regularizers.l2(0.0015), activation='softmax', name='Fc_Out')(Fc_Model)

    model = Model(inputs=model_input, outputs=[Fc_Out])
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.Adam(lr=1e-4),
                  metrics=['accuracy'])

    model.fit({'V_train': V_Dataset, 'I_train': I_Dataset, 'Ir_T_train': Ir_T_Data}, {'Fc_Out': Label_Onehot},
              epochs=200,
              batch_size=32,
              validation_split=0.2)

    model.save('model2.h5')





if __name__ == '__main__':
    main()

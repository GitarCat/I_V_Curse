import numpy as np
import pandas as pd
import keras
import random
from keras import Input
from keras import Model
from keras import regularizers
from keras import initializers
from keras import optimizers
from keras.layers import Dense, Conv1D, Dropout, concatenate, Flatten, Activation, Add, GRU
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import MaxPooling1D, AveragePooling1D
from keras.initializers import glorot_uniform
import scipy.io as sio

np.random.seed(5)

def res_GRU(x, input_units, output_units):
    res_x = GRU(units=output_units)(x)
    res_x = BatchNormalization()(res_x)
    res_x = Activation('relu')(res_x)
    res_x = GRU(units=output_units)(res_x)
    res_x = BatchNormalization()(res_x)
    res_x = Activation('relu')(res_x)
    res_x = GRU(units=output_units)(res_x)
    res_x = BatchNormalization()(res_x)
    res_x = Activation('relu')(res_x)

    if input_units == output_units:
        identity = x
    else:
        identity = res_x = GRU(units=output_units)(x)
    output = Add()([identity, res_x])
    output = Activation('relu')(output)
    return output


##自己看着网上写的
def res_Conv_block(x, k_size, input_filter, output_filter):
    res_x = Conv1D(kernel_size=1, filters=output_filter, strides=1, padding='valid', kernel_initializer=glorot_uniform(seed=0))(x)
    res_x = BatchNormalization()(res_x)
    res_x = Activation('relu')(res_x)
    res_x = Conv1D(kernel_size=k_size, filters=output_filter, strides=1, padding='same', kernel_initializer=glorot_uniform(seed=0))(res_x)
    res_x = BatchNormalization()(res_x)
    res_x = Activation('relu')(res_x)
    res_x = Conv1D(kernel_size=1, filters=output_filter, strides=1, padding='valid', kernel_initializer=glorot_uniform(seed=0))(res_x)
    res_x = BatchNormalization()(res_x)

    if input_filter == output_filter:
        identity = x
    else:
        identity = Conv1D(kernel_size=1, filters=output_filter, strides=1, padding='same')(x)

    x = Add()([identity, res_x])
    output = Activation('relu')(x)
    return output

##网上找的
def identity_block(X, k_size, filters, stage, block):
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters
    # Save the input value. You'll need this later to add back to the main path.
    X_shortcut = X
    # First component of main path
    X = Conv1D(filters=F1, kernel_size=1, strides=1, padding='valid', name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    # Second component of main path
    X = Conv1D(filters=F2, kernel_size=k_size, strides=1, padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)
    # Third component of main path
    X = Conv1D(filters=F3, kernel_size=1, strides=1, padding='valid', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(name=bn_name_base + '2c')(X)
    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X


def convolutional_block(X, k_size, filters, stage, block, s=2):
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters
    # Save the input value
    X_shortcut = X

    ##### MAIN PATH #####
    # First component of main path
    X = Conv1D(filters=F1, kernel_size=1, strides=s, padding='valid', name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    # Second component of main path
    X = Conv1D(filters=F2, kernel_size=k_size, strides=1, padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)
    # Third component of main path
    X = Conv1D(filters=F3, kernel_size=1, strides=1, padding='valid', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(name=bn_name_base + '2c')(X)

    ##### SHORTCUT PATH ####
    X_shortcut = Conv1D(filters=F3, kernel_size=1, strides=s, padding='valid', name=conv_name_base + '1',
                        kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(name=bn_name_base + '1')(X_shortcut)
    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    return X

def main():
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

    # shuffle
    index = np.arange(0, len(label))

    random.shuffle(index)

    # label data process
    label = label.reshape(-1)
    label = np_utils.to_categorical(label, num_classes=13)
    label = label[index]

    # Irr&T data process
    IrrT_data = IrrT_data.reshape((IrrT_data.shape[0], IrrT_data.shape[1]))
    IrrT_data[:, 0] = IrrT_data[:, 0] / 775.29
    IrrT_data[:, 1] = IrrT_data[:, 1] / 51.98
    IrrT_data = IrrT_data[index]

    # V&I dataprocess
    V_data = np.array(V_data, dtype='float32') / (13 * 37.78)
    I_data = np.array(I_data, dtype='float32') / 9.12
    V_data = V_data.reshape((V_data.shape[0], V_data.shape[1], 1))
    I_data = I_data.reshape((I_data.shape[0], I_data.shape[1], 1))
    V_data = V_data[index]
    I_data = I_data[index]


    input_V = Input(shape=(1000, 1), name='input_V')
    input_I = Input(shape=(1000, 1), name='input_I')
    input_IrrT = Input(shape=(2,), name='input_IrrT')
    model_input = [input_V, input_I, input_IrrT]

    IV_train = concatenate([input_V, input_I])

    Conv_Model = Conv1D(filters=32, kernel_size=2, activation='relu',
                 kernel_initializer=initializers.he_normal(), name='Conv1')(IV_train)
    Conv_Model = BatchNormalization(name='bn_conv1')(Conv_Model)
    Conv_Model = MaxPooling1D(pool_size=2, strides=2)(Conv_Model)

##网上找的
    # X = convolutional_block(Conv_Model, k_size=3, filters=[32, 32, 64], stage=2, block='a', s=1)
    # X = identity_block(X, 3, [32, 32, 64], stage=2, block='b')
    # X = identity_block(X, 3, [32, 32, 64], stage=2, block='c')
    # # Stage 3
    # X = convolutional_block(X, k_size=3, filters=[64, 64, 256], stage=3, block='a', s=2)
    # X = identity_block(X, 3, [64, 64, 256], stage=3, block='b')
    # X = identity_block(X, 3, [64, 64, 256], stage=3, block='c')
    # X = identity_block(X, 3, [64, 64, 256], stage=3, block='d')
    # # Stage 4
    # X = convolutional_block(X, k_size=3, filters=[128, 128, 512], stage=4, block='a', s=2)
    # X = identity_block(X, 3, [128, 128, 512], stage=4, block='b')
    # X = identity_block(X, 3, [128, 128, 512], stage=4, block='c')
    # X = identity_block(X, 3, [128, 128, 512], stage=4, block='d')
    # X = identity_block(X, 3, [128, 128, 512], stage=4, block='e')
    # X = identity_block(X, 3, [128, 128, 512], stage=4, block='f')
    # # Stage 5
    # X = convolutional_block(X, k_size=3, filters=[256, 256, 1024], stage=5, block='a', s=2)
    # X = identity_block(X, 3, [256, 256, 1024], stage=5, block='b')
    # X = identity_block(X, 3, [256, 256, 1024], stage=5, block='c')
    #
    # X = AveragePooling1D(pool_size=2)(X)

###自己搭的感觉效果还行，就是过拟合有点严重，本来只在输出添加池化的，结果显存不够，只好在中间插入池化层来减小显存占用
    X = res_Conv_block(Conv_Model, k_size=9, input_filter=32, output_filter=64)
    X = MaxPooling1D(pool_size=2, strides=2)(X)
    X = res_Conv_block(X, k_size=7, input_filter=64, output_filter=128)
    X = MaxPooling1D(pool_size=2, strides=2)(X)
    X = res_Conv_block(X, k_size=5, input_filter=128, output_filter=256)
    X = MaxPooling1D(pool_size=2, strides=2)(X)
    X = res_Conv_block(X, k_size=3, input_filter=256, output_filter=512)
    X = MaxPooling1D(pool_size=2, strides=2)(X)

###
    # X = Conv1D(filters=64, kernel_size=4, activation='relu',
    #                     kernel_initializer=initializers.he_normal())(Conv_Model)
    # X = MaxPooling1D(pool_size=2, strides=2)(X)
    # X = Conv1D(filters=128, kernel_size=8, activation='relu',
    #                     kernel_initializer=initializers.he_normal())(X)
    # X = MaxPooling1D(pool_size=2, strides=2)(X)
    # X = Conv1D(filters=256, kernel_size=6, activation='relu',
    #                     kernel_initializer=initializers.he_normal())(X)
    # X = MaxPooling1D(pool_size=2, strides=2)(X)
    # X = Conv1D(filters=512, kernel_size=6, activation='relu',
    #                     kernel_initializer=initializers.he_normal())(X)
    # X = MaxPooling1D(pool_size=2, strides=2)(X)
    # X = BatchNormalization(epsilon=1e-06, momentum=0.9)(X)
    X = BatchNormalization(epsilon=1e-06, momentum=0.9)(X)
    X = Dropout(0.7)(X)
    X = Flatten()(X)

    X = concatenate([X, input_IrrT], axis=1)

    Fc_Model = Dense(128, activation='relu')(X)
    Fc_Model = Dense(64, activation='relu')(Fc_Model)
    Fc_Out = Dense(13, kernel_regularizer=regularizers.l2(0.0015), activation='softmax', name='Fc_Out')(Fc_Model)

    model = Model(inputs=model_input, outputs=[Fc_Out])
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.Adam(lr=1e-4),
                  metrics=['accuracy'])

    model.fit({'input_V': V_data, 'input_I': I_data, 'input_IrrT': IrrT_data}, {'Fc_Out': label},
              epochs=200,
              batch_size=32,
              validation_split=0.2)

    model.save('CNNonly.h5')

if __name__ == '__main__':
    main()

import numpy as np
import pandas as pd
import keras
import random


from keras import Input
from keras import Model
from keras import regularizers
from keras import initializers
from keras import optimizers
from keras.layers import Dense, Conv1D, Dropout, concatenate
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import MaxPooling1D

import scipy.io as sio


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


def main():
    I_Dataset = pd.read_csv('./dataset/I_data.csv').values
    V_Dataset = pd.read_csv('./dataset/V_Data.csv').values
    Ir_T_Data = pd.read_csv('./dataset/Ir_T.csv').values
    Label = pd.read_csv('./dataset/label.csv').transpose()


if __name__ == '__main__':
    main()
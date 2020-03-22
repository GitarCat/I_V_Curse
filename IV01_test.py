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


def main():
    I_Dataset = pd.read_csv('./dataset/I_data.csv').transpose()
    V_Dataset = pd.read_csv('./dataset/V_Data.csv')
    Ir_T_Data = pd.read_csv('./dataset/Ir_T.csv')
    print(I_Dataset.describe())
    print(Ir_T_Data.describe())


if __name__ == '__main__':
    main()
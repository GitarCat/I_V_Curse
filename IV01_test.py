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
    print(np.random.random(100))


if __name__ == '__main__':
    main()
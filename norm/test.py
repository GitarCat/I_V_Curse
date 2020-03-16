import numpy as np
np.random.seed(7)
#matplotlib inline
import matplotlib.pyplot as plt
import itertools
import h5py
import time
start = time.clock()
from keras.models import load_model
import scipy.io as sio
import random

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
V_data=sio.loadmat('V_test.mat')
V_data=V_data['V_test']
V_data=np.array(V_data)
I_data=sio.loadmat('I_test.mat')
I_data=I_data['I_test']
I_data=np.array(I_data)
IrrT_data=sio.loadmat('IrrT_test.mat')
IrrT_data=IrrT_data['IrrT_test']
IrrT_data=np.array(IrrT_data)
label=sio.loadmat('label_test.mat')
label=label['label_test']

#shuffle
num=len(label)
index = [i for i in range(num)]
random.shuffle(index)
index = np.array(index)

#Irr&T data process
IrrT_data=IrrT_data.reshape((IrrT_data.shape[0],IrrT_data.shape[1]))
IrrT_data=IrrT_data[index]

#V&I dataprocess
V_data=NormalizationV(V_data)
I_data=NormalizationI(I_data)
V_data=V_data.reshape((V_data.shape[0],V_data.shape[1],1))
I_data=I_data.reshape((I_data.shape[0],I_data.shape[1],1))
V_data=V_data[index]
I_data=I_data[index]

#label data process
label=label.reshape(-1)
label=np.array(label)
label=label[index]

model = load_model('2yue26_model.h5')
a=0
y_probability = model.predict({'input_V':V_data,'input_I':I_data,'input_IrrT':IrrT_data})
predict=np.argmax(y_probability,axis=1)
for i in range(len(V_data)):
   print ("%d,%s,%d"%(i,y_probability[i],predict[i]))
for i in range(len(V_data)):
    if label[i]==predict[i]:
        a=a+1

b=a/len(V_data)
print(b)

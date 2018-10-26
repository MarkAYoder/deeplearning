#!/usr/bin/env python3
# LeNet_MNIST-GPU.py 

# GPU Code (Put the following code at beginning of your file)
# Following code is need to allow multiple people to share a GPU.
# This code limits TensorFlow's use of GPU memory to 0.2 of the maximum.
# Please don't be greedy with GPU memory until your final production run.

####################################################################
import tensorflow as tf         
from keras import backend as K  # needed for mixing TensorFlow and Keras commands 
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
sess = tf.Session(config=config)
K.set_session(sess)
####################################################################

# Following shell command runs code on a specific GPU card (card 3) and saves output to a log file
#   CUDA_VISIBLE_DEVICES=3 python LeNet_MNIST-GPU.py > LeNet_MNIST-GPU.log &

# Use the following shell commands to check on load
#   nvidia-smi -l
#   top

import numpy as np
import pandas as pd
from   keras.models import Sequential
#from   keras.models import load_model
from   keras.layers.core import Dense, Activation, Flatten
from   keras.layers.convolutional import Conv2D, MaxPooling2D
from   keras.optimizers import Adam
from   keras.callbacks import EarlyStopping
from   time import time
#import matplotlib.pyplot as plt
#%matplotlib inline

path_to_data = '/work/MA490_DeepLearning/Data'

def BuildLeNet(model,input_shape=(32,32,3),outputs=10):
    model.add(Conv2D(20,5,padding='same',input_shape=input_shape,activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(50,5,padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    model.add(Flatten())
    model.add(Dense(500,activation='relu'))
    model.add(Dense(outputs,activation='softmax'))
    return model

# load training MNIST data
MNIST = np.load(path_to_data+'/MNIST/MNIST_train_40000.npz')
images = MNIST['train_images']
labels = MNIST['train_labels']
print(images.shape)
print(labels.shape)

# define training features and target
X = np.expand_dims(images,-1)
P = pd.get_dummies(
    pd.DataFrame(labels,columns=['digits'],dtype='category')).values
print('X shape',X.shape)
print('P shape',P.shape)

# build LeNet
model = Sequential()
model = BuildLeNet(model,input_shape=(28,28,1),outputs=10)
model.summary()

# run training
epochs = 100
patience = 100 
model.compile(loss='categorical_crossentropy',
        optimizer='Adam',
        metrics=['accuracy'])

time_start = time()
hist = model.fit(X,P,epochs=epochs,validation_split=0.2,verbose=0,
                 batch_size=1000,
                callbacks=[EarlyStopping(patience=patience)])
time_stop  = time()
time_elapsed = time_stop - time_start

results = pd.DataFrame()
results['epoch']           = hist.epoch
results['epoch']           = results['epoch'] + 1
results['training loss']   = hist.history['loss']
results['validation loss'] = np.sqrt(hist.history['val_loss'])
results['training acc']    = hist.history['acc']
results['validation acc']  = np.sqrt(hist.history['val_acc'])

ix = results['validation loss'].idxmin()
ce_training   = results['training loss'].iloc[ix]
ce_validation = results['validation loss'].iloc[ix]
acc_training   = results['training acc'].iloc[ix]
acc_validation = results['validation acc'].iloc[ix]
print('elapsed time',np.round(time_elapsed/60),'min')
print()
print('minimum validation loss index',ix,'of',epochs)
print('cross-entropy')
print('        training =',ce_training)
print('      validation =',ce_validation)
print('accuracy rate')
print('        training =',acc_training)
print('      validation =',acc_validation)

# save model and results
model.save('./LeNet-MNIST-GPU.h5')
results.to_csv('./LeNet-MNIST-GPU.csv',index=False)


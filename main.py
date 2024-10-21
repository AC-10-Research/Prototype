import random
import collections
import glob

import numpy as np
from numpy.matlib import repmat
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM
import pandas as pd

import pretty_midi
#import fluidsynth

# dataset parameters
copies = 200       # number of times to copy the song
window_length = 12  # number of notes used to predict the next note
# neural network parameters
nn_nodes = 10      # number of nodes in the RNN
# training parameters
epochs = 200         # number of epochs used for training
batch_size = None   # size of each batch (None is default)

def dataset_from_song(song,copies,window_length):
    # repeat song "copies" times
    songs = repmat(song,1,copies)[0]
    # number of windows used
    num_windows = len(songs) - window_length

    x_train,y_train = [],[]
    for i in range(num_windows):
        # get a "window_length" number of notes
        x_train.append(songs[i:i+window_length])
        # get the note after the window
        y_train.append(songs[i+window_length])

    # convert to numpy arrays
    x_train = np.array(x_train,dtype='float32')
    x_train = np.expand_dims(x_train,axis=-1)
    y_train = np.array(y_train,dtype='float32')
    y_train = np.expand_dims(y_train,axis=-1)

    return x_train,y_train

# a scale
song = np.array([72,74,76,77,79,81,83,84])
# generate a dataset from copies of the song
x_train,y_train = dataset_from_song(song,copies,window_length)

# specify the architecture of the neural network
model = Sequential()
model.add(SimpleRNN(nn_nodes,activation='relu'))
model.add(Dense(1,activation=None))

# setup the neural network
model.compile(
    loss='MeanSquaredError',
    optimizer='Adam',
    metrics=[])
# use this to save the best weights found so far
# AC: can we set this so that the loss has to be less than 0.001
callback = tf.keras.callbacks.EarlyStopping(
    monitor='loss',patience=10,restore_best_weights=True)


# train the neural network from data
history = model.fit(x_train,y_train,
                    batch_size=batch_size,epochs=epochs,
                    callbacks=[callback])
print("finished training:")
model.evaluate(x_train,y_train)

predictions = model.predict(x_train).round()
correct = sum(predictions == y_train)
N = len(predictions)
print("train set accuracy: %.4f%% (%d/%d)" % (100*correct/N,correct,N))

# shift the scale up by 3 half-steps
test_song = song + 3
# reverse the scale
#test_song = song[-1::-1]
x_test,y_test = dataset_from_song(test_song,3,window_length)
predictions = model.predict(x_test).round()
correct = sum(predictions == y_test)
N = len(predictions)
print(" test set accuracy: %.4f%% (%d/%d)" % (100*correct/N,correct,N))

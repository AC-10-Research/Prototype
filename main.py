import numpy as np
import tensorflow as tf
from numpy.matlib import repmat
from tensorflow.keras.layers import Dense, SimpleRNN
from tensorflow.keras.models import Sequential
import pretty_midi
import fluidsynth

# dataset parameters
copies = 200       # number of times to copy the song
window_length = 12  # number of notes used to predict the next note
# neural network parameters
nn_nodes = 10      # number of nodes in the RNN
# training parameters
epochs = 200         # number of epochs used for training
batch_size = None   # size of each batch (None is default)

def create_midi_from_predictions(predictions, output_file='output.mid', instrument_program='0'):
    midi_data = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=instrument_program)
    start_time = 0.0
    duration = 0.5

    for note in predictions:
        note_number = int(note)
        midi_note = pretty_midi.Note(velocity=100, pitch=note_number, start=start_time, end=start_time + duration)
        instrument.notes.append(midi_note)
        start_time += duration

    midi_data.instruments.append(instrument)
    midi_data.write(output_file)
    print(f'MIDI file saved as: {output_file}')

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

#  Middle C Scale
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
create_midi_from_predictions(predictions, output_file='predicted_song.mid', instrument_program=0)

pm = pretty_midi.PrettyMIDI('predicted_song.mid')
def display_audio(pm: pretty_midi.PrettyMIDI, seconds = 30):
    waveform = pm.fluidsynth(fs=_SAMPLING_RATE)
    waveform_short = waveform[:seconds*_SAMPLING_RATE]
    return display.Audio(waveform_short, rate=_SAMPLING_RATE)
display_audio(pm)

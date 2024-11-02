import random
import numpy as np
import tensorflow as tf
from numpy.matlib import repmat
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

COPIES = 200           # Number of times to copy the song to create a larger dataset
WINDOW_LENGTH = 12     # Number of notes per input sequence (sequence length)

NN_NODES = 15          # Number of nodes in the RNN layer
EPOCHS = 100           # Number of training epochs
BATCH_SIZE = None      # Default batch size (handled by TensorFlow internally)


def dataset_from_song(song, copies, window_length):
    """Generate training data from repeated copies of the song.
    
    Args:
        song (np.array): The original sequence of notes.
        copies (int): Number of times to repeat the song.
        window_length (int): Length of the input sequence window.

    Returns:
        tuple: Input (x_train) and target (y_train) datasets as NumPy arrays.
    """
    # Repeat the song `copies` times to create more data points
    repeated_song = repmat(song, 1, copies)[0]
    num_windows = len(repeated_song) - window_length

    # Create input (X) and target (Y) datasets
    x_train, y_train = [], []
    for i in range(num_windows):
        x_train.append(repeated_song[i:i + window_length])  # Input sequence
        y_train.append(repeated_song[i + window_length])    # Next note to predict

    # Convert to NumPy arrays and reshape for RNN input
    x_train = np.expand_dims(np.array(x_train, dtype='float32'), axis=-1)  # Shape: (num_samples, window_length, 1)
    y_train = np.expand_dims(np.array(y_train, dtype='float32'), axis=-1)  # Shape: (num_samples, 1)

    return x_train, y_train


# Define the song and generate the training data
song = np.array([72, 74, 76, 77, 79, 81, 83, 84])  # MIDI note numbers representing a simple melody
x_train, y_train = dataset_from_song(song, COPIES, WINDOW_LENGTH)

# Build the RNN model
model = Sequential([
    SimpleRNN(NN_NODES, activation='relu', input_shape=(WINDOW_LENGTH, 1)),  # RNN layer with ReLU activation
    Dense(1)  # Single output for predicting the next note in the sequence
])

# Compile the model
model.compile(
    loss='mean_squared_error',  # Loss function to minimize the difference between predicted and true values
    optimizer='adam'            # Adam optimizer for efficient training
)

callback = tf.keras.callbacks.EarlyStopping(
    monitor='loss', patience=100, restore_best_weights=True  # Stop training if no improvement for 100 epochs
)

# Train the model
history = model.fit(
    x_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[callback]
)

print("Finished training:")
loss = model.evaluate(x_train, y_train)
print(f"Training loss: {loss:.4f}")

predictions = model.predict(x_train).round()  # Round predictions to nearest integer (MIDI note number)
accuracy = 100 * np.mean(predictions == y_train)  # Calculate accuracy as a percentage
print(f"Train set accuracy: {accuracy:.2f}%")

accuracies = []
for scale_steps in range(1, 7, 2):
    test_song = song + scale_steps  # Transpose the song by adding `scale_steps` to each note
    x_test, y_test = dataset_from_song(test_song, copies=3, window_length=WINDOW_LENGTH)

    predictions = model.predict(x_test).round()  # Make predictions and round to nearest integer
    acc = 100 * np.mean(predictions == y_test)  # Calculate accuracy for the test set
    accuracies.append(acc)

    print(f"Test set accuracy for scale step {scale_steps}: {acc:.2f}%")

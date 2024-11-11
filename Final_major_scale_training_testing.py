## Training/Testing on major scale
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from numpy.matlib import repmat
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from itertools import islice

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Dataset parameters
COPIES = 200           # Number of times to copy the song
WINDOW_LENGTH = 12      # Number of notes per input sequence

# Neural network parameters
NN_NODES = 15           # Number of nodes in the RNN
EPOCHS = 100            # Number of training epochs
BATCH_SIZE = None       # Default batch size (auto-handled by TensorFlow)

# Major scales from your scale maker code
major_scales = {
    'C_major': [60, 62, 64, 65, 67, 69, 71, 72],
    'C#_major': [61, 63, 65, 66, 68, 70, 72, 73],
    'D_major': [62, 64, 66, 67, 69, 71, 73, 74],
    'Eb_major': [63, 65, 67, 68, 70, 72, 74, 75],
    'E_major': [64, 66, 68, 69, 71, 73, 75, 76],
    'F_major': [65, 67, 69, 70, 72, 74, 76, 77],
    'F#_major': [66, 68, 70, 71, 73, 75, 77, 78],
    'G_major': [67, 69, 71, 72, 74, 76, 78, 79],
    'Ab_major': [68, 70, 72, 73, 75, 77, 79, 80],
    'A_major': [69, 71, 73, 74, 76, 78, 80, 81],
    'Bb_major': [70, 72, 74, 75, 77, 79, 81, 82],
    'B_major': [71, 73, 75, 76, 78, 80, 82, 83]
}

def dataset_from_song(song, copies, window_length):
    # Repeat the song `copies` times to create more data points
    repeated_song = repmat(song, 1, copies)[0]
    num_windows = len(repeated_song) - window_length

    # Create input (X) and target (Y) datasets
    x_train, y_train = [], []
    for i in range(num_windows):
        x_train.append(repeated_song[i:i + window_length])
        y_train.append(repeated_song[i + window_length])

    # Convert to NumPy arrays and reshape for RNN input
    x_train = np.expand_dims(np.array(x_train, dtype='float32'), axis=-1)
    y_train = np.expand_dims(np.array(y_train, dtype='float32'), axis=-1)

    return x_train, y_train

x_train_all, y_train_all =[],[]
# Define the song and generate the training data
for key, song in islice(major_scales.items(), 9):
    x_train, y_train = dataset_from_song(song[1], COPIES, WINDOW_LENGTH)
    x_train_all.append(x_train)
    y_train_all.append(y_train)

x_test_all, y_test_all =[],[]
last_three_items = list(major_scales.items())[-3:]
# Define the song and generate the training data
for key, song in last_three_items:
    x_test, y_test = dataset_from_song(song[1], COPIES, WINDOW_LENGTH)
    x_test_all.append(x_test)
    y_test_all.append(y_test)
    
# Build the RNN model
model = Sequential([
    SimpleRNN(NN_NODES, activation='relu', input_shape=(WINDOW_LENGTH, 1)),
    Dense(1)  # Single output for predicting the next note
])

# Compile the model
model.compile(
    loss='mean_squared_error',
    optimizer='adam'
)

# Early stopping callback to prevent overfitting
callback = tf.keras.callbacks.EarlyStopping(
    monitor='loss', patience=100, restore_best_weights=True
)

# Train the model
history = model.fit(
    x_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[callback]
)

# Evaluate the model on the training set
print("Finished training:")
model.evaluate(x_train, y_train)

# Make predictions on the training set
predictions = model.predict(x_train).round()
accuracy = 100 * np.mean(predictions == y_train)
print(f"Train set accuracy: {accuracy:.2f}%")

# Plot training predictions vs. actual values
def plot_predictions(predictions, y_true, title, filename):
    """Plot predictions vs. true values and save the figure."""
    plt.figure()
    index = np.arange(len(predictions))
    plt.plot(index, predictions, 'b', label='Predictions')
    plt.plot(index, y_true, 'r', label='True')
    plt.scatter(index, predictions, color='b')
    plt.scatter(index, y_true, color='r')
    plt.xlabel("Datapoint")
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

plot_predictions(predictions[:50], y_train[:50], "Training Set Predictions", "Training.png")

# Plot training loss per epoch
plt.figure()
plt.plot(history.history['loss'], marker='o', linestyle='-', color='b')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss per Epoch')
plt.grid(True)
plt.savefig("Training_error.png")
plt.close()


# Test the model with the rest of patterns
accuracies = []
for index in range(0, 3):
    x_test, y_test = x_test_all[index], y_test_all[index]

    predictions = model.predict(x_test).round()
    acc = 100 * np.mean(predictions == y_test)
    accuracies.append(acc)

    print(f"Test set accuracy for datapoint {index}: {acc:.2f}%")
    plot_predictions(predictions, y_test, 
                     f"Test Set (Index {index})", f"Testing_{index}.png")

# Plot test accuracies for different scale steps
plt.figure(figsize=(10, 5))
plt.bar(range(len(accuracies)), accuracies, color='skyblue', width=0.5)
plt.xlabel('Test Case')
plt.ylabel('Accuracy (%)')
plt.title('Test Set Accuracy for Different Scale Steps')
plt.grid(axis='y')
plt.savefig("Testing_accuracies.png")
plt.close()

# Suppress TensorFlow informational messages
'''
    Added Logistic Regression (LR) model for comparison
    Retained Federated DNN model
'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Only show warnings and errors

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import SGD
import tensorflow_federated as tff
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load MNIST data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Preprocess data for DNN
x_train_dnn = x_train / 255.0  # Normalize pixel values
x_test_dnn = x_test / 255.0

# Preprocess data for Logistic Regression (flattened input)
x_train_lr = x_train.reshape(-1, 28 * 28) / 255.0
x_test_lr = x_test.reshape(-1, 28 * 28) / 255.0

# Federated Learning Preprocessing for DNN
x_train_fed = np.expand_dims(x_train_dnn, axis=-1)
x_test_fed = np.expand_dims(x_test_dnn, axis=-1)

NUM_CLIENTS = 100
client_data_size = len(x_train_fed) // NUM_CLIENTS
federated_train_data = []

for i in range(NUM_CLIENTS):
    start = i * client_data_size
    end = start + client_data_size
    client_x = x_train_fed[start:end]
    client_y = y_train[start:end]
    federated_train_data.append((client_x, client_y))

# Define the DNN model
def create_dnn_model():
    model = Sequential([
        Flatten(input_shape=(28, 28, 1)),  # Flatten 2D images into 1D vectors
        Dense(128, activation='relu'),    # First hidden layer with 128 units
        Dense(64, activation='relu'),     # Second hidden layer with 64 units
        Dense(10, activation='softmax')  # Output layer with 10 units (one per class)
    ])
    return model

# Convert to TFF model
def model_fn():
    keras_model = create_dnn_model()
    return tff.learning.from_keras_model(
        keras_model,
        input_spec=(
            tf.TensorSpec(shape=[None, 28, 28, 1], dtype=tf.float32),
            tf.TensorSpec(shape=[None], dtype=tf.int32)
        ),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )

# Preprocess federated data
def preprocess_for_federated_learning(client_x, client_y):
    client_x = client_x.astype(np.float32)
    client_y = client_y.astype(np.int32)
    dataset = tf.data.Dataset.from_tensor_slices((client_x, client_y))
    return dataset.shuffle(100).batch(50)

federated_train_datasets = [
    preprocess_for_federated_learning(client_x, client_y)
    for client_x, client_y in federated_train_data
]

# Define Federated Averaging Process
iterative_process = tff.learning.build_federated_averaging_process(
    model_fn,
    client_optimizer_fn=lambda: SGD(learning_rate=0.01),
    server_optimizer_fn=lambda: SGD(learning_rate=1.0)
)

# Initialize the federated learning process
state = iterative_process.initialize()

# Perform federated training
NUM_ROUNDS = 100  # Number of communication rounds
for round_num in range(NUM_ROUNDS):
    state, metrics = iterative_process.next(state, federated_train_datasets)
    print(f"Round {round_num+1}, Metrics: {metrics}")

# Convert TFF model state to Keras model
final_keras_model = create_dnn_model()
state.model.assign_weights_to(final_keras_model)

# Evaluate the federated DNN model
final_keras_model.compile(optimizer=SGD(learning_rate=0.01),
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])

test_loss, test_accuracy = final_keras_model.evaluate(x_test_fed, y_test, verbose=0)
print(f"Federated DNN Model Accuracy: {test_accuracy:.4f}")

# Logistic Regression Model
lr_model = LogisticRegression(max_iter=100)
lr_model.fit(x_train_lr, y_train)

# Predict and evaluate Logistic Regression model
y_pred_lr = lr_model.predict(x_test_lr)
lr_accuracy = accuracy_score(y_test, y_pred_lr)
print(f"Logistic Regression Model Accuracy: {lr_accuracy:.4f}")
# Suppress TensorFlow informational messages
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
from sklearn.preprocessing import StandardScaler

# Load MNIST data (60,000 training samples, 10,000 test samples)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Check if the dataset is loaded correctly (expecting (60000, 28, 28) for x_train and x_test)
if x_train.shape[0] != 60000 or y_train.shape[0] != 60000:
    raise ValueError("The MNIST training dataset didn't load correctly. Please check your data.")
if x_test.shape[0] != 10000 or y_test.shape[0] != 10000:
    raise ValueError("The MNIST test dataset didn't load correctly. Please check your data.")

# Preprocess data for DNN
x_train_dnn = x_train / 255.0  # Normalize pixel values
x_test_dnn = x_test / 255.0

# Federated Learning Preprocessing for DNN
x_train_fed = np.expand_dims(x_train_dnn, axis=-1)  # Add channel dimension
x_test_fed = np.expand_dims(x_test_dnn, axis=-1)    # Add channel dimension

# Create a TensorFlow Dataset for testing (reshape correctly)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test_fed, y_test))

# Ensure the test dataset has the correct shape (batch_size, 28, 28, 1)
test_dataset = test_dataset.map(lambda x, y: (tf.expand_dims(x, -1), y))

# Batch the test dataset
test_dataset = test_dataset.batch(32)

# Define the DNN model
def create_dnn_model():
    model = Sequential([
        Flatten(input_shape=(28, 28, 1)),  # Ensure the input has 3 dimensions (28x28x1)
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

# Federated Learning Preprocessing
NUM_CLIENTS = 100
client_data_size = len(x_train_fed) // NUM_CLIENTS
federated_train_data = []

for i in range(NUM_CLIENTS):
    start = i * client_data_size
    end = start + client_data_size
    client_x = x_train_fed[start:end]
    client_y = y_train[start:end]
    federated_train_data.append((client_x, client_y))

def preprocess_for_federated_learning(client_x, client_y):
    client_x = client_x.astype(np.float32)
    client_y = client_y.astype(np.int32)
    dataset = tf.data.Dataset.from_tensor_slices((client_x, client_y))
    return dataset.shuffle(100).batch(100)  # Increase batch size to 100

federated_train_datasets = [
    preprocess_for_federated_learning(client_x, client_y)
    for client_x, client_y in federated_train_data
]

# Define Federated Averaging Process
def client_optimizer_fn():
    return SGD(learning_rate=0.005)  # Adjusted learning rate

def server_optimizer_fn():
    return SGD(learning_rate=1.0)

iterative_process = tff.learning.build_federated_averaging_process(
    model_fn,
    client_optimizer_fn=client_optimizer_fn,  # Pass the function, not the optimizer
    server_optimizer_fn=server_optimizer_fn   # Pass the function, not the optimizer
)

# Initialize the federated learning process
state = iterative_process.initialize()

# Perform federated training
NUM_ROUNDS = 100  # Increased number of rounds for better convergence
for round_num in range(NUM_ROUNDS):
    state, metrics = iterative_process.next(state, federated_train_datasets)
    print(f"Round {round_num+1}, Metrics: {metrics}")

# Convert TFF model state to Keras model
final_keras_model = create_dnn_model()
state.model.assign_weights_to(final_keras_model)

# Compile the Keras model before evaluation
final_keras_model.compile(optimizer=SGD(learning_rate=0.01),
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])

# Evaluate the federated DNN model
test_accuracy = final_keras_model.evaluate(test_dataset, verbose=0)[1]
test_error_rate = 1 - test_accuracy
print(f"Federated DNN Model Accuracy: {test_accuracy:.4f}")
print(f"Federated DNN Model Error Rate: {test_error_rate:.4f}")

# Preprocessing for Logistic Regression
x_train_lr = x_train.reshape(-1, 28 * 28) / 255.0  # Flatten for LR
x_test_lr = x_test.reshape(-1, 28 * 28) / 255.0

# Scale the data for Logistic Regression
scaler = StandardScaler()
x_train_lr_scaled = scaler.fit_transform(x_train_lr)
x_test_lr_scaled = scaler.transform(x_test_lr)

# Train Logistic Regression model
lr_model = LogisticRegression(solver='liblinear', max_iter=100)  # Increased max_iter
lr_model.fit(x_train_lr_scaled, y_train)

# Predict and evaluate Logistic Regression model
y_pred_lr = lr_model.predict(x_test_lr_scaled)
lr_accuracy = accuracy_score(y_test, y_pred_lr)
lr_error_rate = 1 - lr_accuracy
print(f"Logistic Regression Model Accuracy: {lr_accuracy:.4f}")
print(f"Logistic Regression Model Error Rate: {lr_error_rate:.4f}")


'''
Federated DNN Model Accuracy: 0.8128
Federated DNN Model Error Rate: 0.1872

Logistic Regression Model Accuracy: 0.9171
Logistic Regression Model Error Rate: 0.0829

'''
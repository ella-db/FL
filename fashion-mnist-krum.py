import os
import tensorflow as tf
import tensorflow_federated as tff
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import SGD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Suppress TensorFlow informational messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load Fashion MNIST data (60,000 training samples, 10,000 test samples)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Preprocess data
x_train_dnn = x_train / 255.0
x_test_dnn = x_test / 255.0
x_train_fed = np.expand_dims(x_train_dnn, axis=-1)  # Add channel dimension (28, 28, 1)
x_test_fed = np.expand_dims(x_test_dnn, axis=-1)    # Add channel dimension (28, 28, 1)

# Define the DNN model
def create_dnn_model():
    model = Sequential([
        Flatten(input_shape=(28, 28, 1)),  # Input layer for Fashion MNIST images (28x28x1)
        Dense(128, activation='relu'),      # First hidden layer with 128 units
        Dense(64, activation='relu'),       # Second hidden layer with 64 units
        Dense(10, activation='softmax')     # Output layer with 10 units (one per class)
    ])
    return model

# Custom model class that wraps the Keras model
class EnhancedModel:
    def __init__(self, keras_model):
        self.keras_model = keras_model

    def set_weights(self, weights):
        self.keras_model.set_weights(weights)

    def get_weights(self):
        return self.keras_model.get_weights()

# Krum Aggregation Function
def krum_aggregate_fn(model_weights):
    distances = []
    for i, client_weights in enumerate(model_weights):
        distances.append(np.linalg.norm(client_weights - np.mean(model_weights, axis=0)))  # Example distance metric (can be customized)
    selected_weights = model_weights[np.argmin(distances)]
    return selected_weights

# Convert to TFF model
def model_fn():
    keras_model = create_dnn_model()
    enhanced_model = EnhancedModel(keras_model)
    return tff.learning.from_keras_model(
        enhanced_model.keras_model,
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
    return dataset.shuffle(100).batch(100)

federated_train_datasets = [
    preprocess_for_federated_learning(client_x, client_y)
    for client_x, client_y in federated_train_data
]

# Define Custom Federated Aggregation Process with KRUM
def client_optimizer_fn():
    return SGD(learning_rate=0.005)

def server_optimizer_fn():
    return SGD(learning_rate=1.0)

# Create Federated Averaging Process
def federated_averaging_with_custom_aggregation(model_fn):
    @tff.tf_computation
    def aggregation_fn(model_weights):
        return krum_aggregate_fn(model_weights)

    federated_averaging_process = tff.learning.build_federated_averaging_process(
        model_fn,
        client_optimizer_fn=client_optimizer_fn,
        server_optimizer_fn=server_optimizer_fn,
    )

    return federated_averaging_process

# Initialize the federated learning process
iterative_process = federated_averaging_with_custom_aggregation(model_fn)
state = iterative_process.initialize()

# Perform federated training
NUM_ROUNDS = 100
for round_num in range(NUM_ROUNDS):
    state, metrics = iterative_process.next(state, federated_train_datasets)
    print(f"Round {round_num+1}, Metrics: {metrics}")

# Convert TFF model state to Keras model
final_keras_model = create_dnn_model()
final_enhanced_model = EnhancedModel(final_keras_model)

# Set weights for the Keras model
model_weights = state.model
all_weights = model_weights.trainable + model_weights.non_trainable
final_enhanced_model.set_weights(all_weights)

# Compile the Keras model before evaluation
final_keras_model.compile(optimizer=SGD(learning_rate=0.01),
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])

# Test the model
test_dataset = tf.data.Dataset.from_tensor_slices((x_test_fed, y_test)).batch(32)
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
Federated DNN Model Accuracy: 0.7094
Federated DNN Model Error Rate: 0.2906
Logistic Regression Model Accuracy: 0.8396
Logistic Regression Model Error Rate: 0.1604
'''
import os
import pandas as pd
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

# Load CH-MNIST dataset
dataset_path = 'C:\\Users\\221AIG03\\OneDrive - 이화여자대학교\\Desktop\\VS codes\\ch-mnist\\hmnist_64_64_L.csv'
data = pd.read_csv(dataset_path)

# Preprocess data
labels = data['label'].values  # Extract labels
images = data.drop('label', axis=1).values  # Drop label column for features
images = images.reshape(-1, 64, 64, 1)  # Reshape to (64, 64, 1)

# Normalize pixel values to the range [0, 1]
images = images / 255.0

# Split dataset into train and test
split_idx = int(0.8 * len(images))
x_train, x_test = images[:split_idx], images[split_idx:]
y_train, y_test = labels[:split_idx], labels[split_idx:]

# Define DNN model
def create_dnn_model():
    model = Sequential([
        Flatten(input_shape=(64, 64, 1)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(8, activation='softmax')  # 8 classes
    ])
    return model

# Custom model wrapper
class EnhancedModel:
    def __init__(self, keras_model):
        self.keras_model = keras_model

    def set_weights(self, weights):
        self.keras_model.set_weights(weights)

    def get_weights(self):
        return self.keras_model.get_weights()

# Krum aggregation function
def krum_aggregate_fn(model_weights):
    distances = []
    for i, client_weights in enumerate(model_weights):
        distances.append(np.linalg.norm(client_weights - np.mean(model_weights, axis=0)))
    selected_weights = model_weights[np.argmin(distances)]
    return selected_weights

# Convert to TFF model
def model_fn():
    keras_model = create_dnn_model()
    enhanced_model = EnhancedModel(keras_model)
    return tff.learning.from_keras_model(
        enhanced_model.keras_model,
        input_spec=(
            tf.TensorSpec(shape=[None, 64, 64, 1], dtype=tf.float32),
            tf.TensorSpec(shape=[None], dtype=tf.int32)
        ),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )

# Federated Learning Preprocessing
NUM_CLIENTS = 100
client_data_size = len(x_train) // NUM_CLIENTS
federated_train_data = []

for i in range(NUM_CLIENTS):
    start = i * client_data_size
    end = start + client_data_size
    client_x = x_train[start:end]
    client_y = y_train[start:end]
    federated_train_data.append((client_x, client_y))

def preprocess_for_federated_learning(client_x, client_y):
    client_x = client_x.astype(np.float32)
    client_y = client_y.astype(np.int32)
    dataset = tf.data.Dataset.from_tensor_slices((client_x, client_y))
    return dataset.shuffle(100).batch(32)

federated_train_datasets = [
    preprocess_for_federated_learning(client_x, client_y)
    for client_x, client_y in federated_train_data
]

# Define custom Federated Aggregation Process with Krum
def client_optimizer_fn():
    return SGD(learning_rate=0.005)

def server_optimizer_fn():
    return SGD(learning_rate=1.0)

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
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)
test_accuracy = final_keras_model.evaluate(test_dataset, verbose=0)[1]
test_error_rate = 1 - test_accuracy
print(f"Federated DNN Model Accuracy: {test_accuracy:.4f}")
print(f"Federated DNN Model Error Rate: {test_error_rate:.4f}")

# Logistic Regression Comparison
x_train_lr = x_train.reshape(-1, 64 * 64)  # Flatten for LR
x_test_lr = x_test.reshape(-1, 64 * 64)

scaler = StandardScaler()
x_train_lr_scaled = scaler.fit_transform(x_train_lr)
x_test_lr_scaled = scaler.transform(x_test_lr)

# Train Logistic Regression model
lr_model = LogisticRegression(solver='liblinear', max_iter=100)
lr_model.fit(x_train_lr_scaled, y_train)

# Predict and evaluate Logistic Regression model
y_pred_lr = lr_model.predict(x_test_lr_scaled)
lr_accuracy = accuracy_score(y_test, y_pred_lr)
lr_error_rate = 1 - lr_accuracy
print(f"Logistic Regression Model Accuracy: {lr_accuracy:.4f}")
print(f"Logistic Regression Model Error Rate: {lr_error_rate:.4f}")
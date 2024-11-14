# Suppress TensorFlow informational messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Only show warnings and errors

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler  # Import StandardScaler
import tensorflow_federated as tff
import numpy as np

# Load MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess the data for DNN
x_train_dnn = x_train / 255.0  # Normalize for DNN
x_test_dnn = x_test / 255.0

# Flatten data for Logistic Regression
x_train_lr = x_train.reshape(-1, 28*28)
x_test_lr = x_test.reshape(-1, 28*28)

# Standard Scaling for Logistic Regression
scaler = StandardScaler()
x_train_lr_scaled = scaler.fit_transform(x_train_lr)
x_test_lr_scaled = scaler.transform(x_test_lr)

# DNN Model
dnn_model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

dnn_model.compile(optimizer=Adam(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

# Train the DNN model
dnn_model.fit(x_train_dnn, y_train, epochs=5, validation_data=(x_test_dnn, y_test))

# Evaluate the DNN model
dnn_loss, dnn_accuracy = dnn_model.evaluate(x_test_dnn, y_test)
print(f"DNN Model Accuracy: {dnn_accuracy:.4f}")

# Logistic Regression Model
lr_model = LogisticRegression(max_iter=200)
lr_model.fit(x_train_lr_scaled, y_train)

# Predict and evaluate Logistic Regression model
y_pred_lr = lr_model.predict(x_test_lr_scaled)
lr_accuracy = accuracy_score(y_test, y_pred_lr)
print(f"Logistic Regression Model Accuracy: {lr_accuracy:.4f}")

# Federated Learning setup
# Split data among clients
NUM_CLIENTS = 10
client_data_size = len(x_train) // NUM_CLIENTS
federated_train_data = []

for i in range(NUM_CLIENTS):
    start = i * client_data_size
    end = start + client_data_size
    client_x = x_train[start:end] / 255.0  # Normalize
    client_y = y_train[start:end]
    federated_train_data.append((client_x, client_y))

# Define Federated Learning model
def create_keras_model():
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])
    return model

def model_fn():
    keras_model = create_keras_model()
    return tff.learning.from_keras_model(
        keras_model,
        input_spec=(
            tf.TensorSpec(shape=[None, 28, 28], dtype=tf.float32),
            tf.TensorSpec(shape=[None], dtype=tf.int32)
        ),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )

# Federated data format
def preprocess_for_federated_learning(client_x, client_y):
    # Convert data types to match TFF expectations
    client_x = client_x.astype(np.float32)
    client_y = client_y.astype(np.int32)
    dataset = tf.data.Dataset.from_tensor_slices((client_x, client_y))
    return dataset.shuffle(100).batch(32)

federated_train_datasets = [
    preprocess_for_federated_learning(client_x, client_y)
    for client_x, client_y in federated_train_data
]

# Define Federated Averaging Process
iterative_process = tff.learning.build_federated_averaging_process(
    model_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02),
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0)
)

# Initialize the federated learning process
state = iterative_process.initialize()

# Perform federated training
NUM_ROUNDS = 10
for round_num in range(NUM_ROUNDS):
    state, metrics = iterative_process.next(state, federated_train_datasets)
    print(f"Round {round_num+1}, Metrics: {metrics}")

# Final evaluation
# Converting TFF model state to Keras model
final_keras_model = create_keras_model()
state.model.assign_weights_to(final_keras_model)

# Evaluate the federated model
test_loss, test_accuracy = final_keras_model.evaluate(x_test_dnn, y_test)
print(f"Federated Model Accuracy: {test_accuracy:.4f}")


#first try DNN: 09767 LR 0.9258
#second try DNN: 0.9750 LR: error (added scaling for data ahead)
#third try DNN: 0.9731 LR: 0.9219
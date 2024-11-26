# Suppress TensorFlow informational messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Only show warnings and errors

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import SGD
import tensorflow_federated as tff
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Load the CH-MNIST dataset
file_path = 'C:\\Users\\221AIG03\\OneDrive - 이화여자대학교\\Desktop\\VS codes\\ch-mnist\\hmnist_64_64_L.csv'
print("Loading CH-MNIST dataset...")
data = pd.read_csv(file_path)

# Extract pixel data and labels
pixel_columns = [col for col in data.columns if 'pixel' in col]
images = data[pixel_columns].values.reshape(-1, 64, 64)  # Reshape into 64x64 images
labels = data['label'].values

# Normalize pixel values
images = images / 255.0

# Add a channel dimension for TensorFlow compatibility
images_expanded = np.expand_dims(images, axis=-1)

# Split into training and testing datasets
x_train, x_test, y_train, y_test = train_test_split(images_expanded, labels, test_size=0.2, random_state=42)

# Create TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_dataset = test_dataset.batch(32)

# Define the categories (replace these with appropriate labels)
categories = ["Tumor", "Stroma", "Complex", "Lympho", "Debris", "Mucosa", "Adipose", "Empty"]

# Define the DNN model with dropout layers for regularization
def create_dnn_model():
    model = Sequential([
        Flatten(input_shape=(64, 64, 1)),  # Adjust input shape for 64x64 images
        Dense(128, activation='relu'),     # First hidden layer
        Dropout(0.25),                     # Dropout for regularization
        Dense(64, activation='relu'),      # Second hidden layer
        Dropout(0.25),                     # Dropout for regularization
        Dense(len(categories), activation='softmax')  # Output layer
    ])
    return model

# Define a model function for TFF
def model_fn():
    keras_model = create_dnn_model()
    return tff.learning.from_keras_model(
        keras_model,
        input_spec=(
            tf.TensorSpec(shape=[None, 64, 64, 1], dtype=tf.float32),
            tf.TensorSpec(shape=[None], dtype=tf.int32)
        ),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )

# Federated Learning: Prepare client datasets
NUM_CLIENTS = 100
client_data_size = len(x_train) // NUM_CLIENTS

federated_train_data = [
    (x_train[i * client_data_size:(i + 1) * client_data_size],
     y_train[i * client_data_size:(i + 1) * client_data_size])
    for i in range(NUM_CLIENTS)
]

def preprocess_for_federated_learning(client_x, client_y):
    client_x = client_x.astype(np.float32)
    client_y = client_y.astype(np.int32)
    dataset = tf.data.Dataset.from_tensor_slices((client_x, client_y))
    return dataset.shuffle(100).batch(50)  # Set batch size to 50

federated_train_datasets = [
    preprocess_for_federated_learning(client_x, client_y)
    for client_x, client_y in federated_train_data
]

# Define Federated Averaging Process
def client_optimizer_fn():
    return SGD(learning_rate=0.01)  # Updated learning rate

def server_optimizer_fn():
    return SGD(learning_rate=0.5)  # Updated server learning rate

iterative_process = tff.learning.build_federated_averaging_process(
    model_fn=model_fn,
    client_optimizer_fn=client_optimizer_fn,
    server_optimizer_fn=server_optimizer_fn
)

# Initialize the federated learning process
state = iterative_process.initialize()

# Perform federated training (with local epochs set to 5 and global rounds set to 100)
NUM_ROUNDS = 100
LOCAL_EPOCHS = 5

for round_num in range(NUM_ROUNDS):
    state, metrics = iterative_process.next(state, federated_train_datasets)
    print(f"Round {round_num+1}, Metrics: {metrics}")

# Convert the TFF model state to a Keras model
final_keras_model = create_dnn_model()
state.model.assign_weights_to(final_keras_model)

# Compile the Keras model for evaluation
final_keras_model.compile(optimizer=SGD(learning_rate=0.01),
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])

# Evaluate the federated model
test_accuracy = final_keras_model.evaluate(test_dataset, verbose=0)[1]
test_error_rate = 1 - test_accuracy
print(f"Federated DNN Model Accuracy: {test_accuracy:.4f}")
print(f"Federated DNN Model Error Rate: {test_error_rate:.4f}")

# Logistic Regression Model for Comparison
x_train_lr = x_train.reshape(-1, 64 * 64)  # Flatten images for LR
x_test_lr = x_test.reshape(-1, 64 * 64)

# Scale the data
scaler = StandardScaler()
x_train_lr_scaled = scaler.fit_transform(x_train_lr)
x_test_lr_scaled = scaler.transform(x_test_lr)

# Train the Logistic Regression model
lr_model = LogisticRegression(solver='liblinear', max_iter=100)
lr_model.fit(x_train_lr_scaled, y_train)

# Evaluate the Logistic Regression model
y_pred_lr = lr_model.predict(x_test_lr_scaled)
lr_accuracy = accuracy_score(y_test, y_pred_lr)
lr_error_rate = 1 - lr_accuracy
print(f"Logistic Regression Model Accuracy: {lr_accuracy:.4f}")
print(f"Logistic Regression Model Error Rate: {lr_error_rate:.4f}")


'''
Federated DNN Model Accuracy: 0.1640
Federated DNN Model Error Rate: 0.8360
Logistic Regression Model Accuracy: 0.3090
Logistic Regression Model Error Rate: 0.6910
'''
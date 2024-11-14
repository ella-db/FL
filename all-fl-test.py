import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.datasets import mnist, fashion_mnist
import tensorflow_datasets as tfds
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow_federated as tff
import numpy as np
import pandas as pd

# Set a shorter directory for TensorFlow datasets
os.environ['TFDS_DATA_DIR'] = 'C:/tfds_data'

# Load all datasets and store them in a dictionary
def load_mnist():
    return mnist.load_data()

def load_fashion_mnist():
    return fashion_mnist.load_data()

def load_ch_mnist():
    chmnist = tfds.load("colorectal_histology", split=["train[:80%]", "train[80%:]"], as_supervised=True)
    train_data, test_data = chmnist
    train_images = [tf.image.rgb_to_grayscale(img) for img, _ in train_data]
    train_labels = [lbl for _, lbl in train_data]
    test_images = [tf.image.rgb_to_grayscale(img) for img, _ in test_data]
    test_labels = [lbl for _, lbl in test_data]
    
    # Convert to numpy arrays and resize
    train_images = tf.image.resize(train_images, (28, 28))
    test_images = tf.image.resize(test_images, (28, 28))
    train_images, test_images = np.array(train_images), np.array(test_images)
    train_labels, test_labels = np.array(train_labels), np.array(test_labels)
    
    # Apply data augmentation to training images
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )
    train_images_augmented = next(datagen.flow(train_images, train_labels, batch_size=len(train_images)))[0]
    
    return (train_images_augmented, train_labels), (test_images, test_labels)

datasets = {
    "MNIST": load_mnist,
    "Fashion_MNIST": load_fashion_mnist,
    "CH_MNIST": load_ch_mnist
}

# Define a more complex DNN model with dropout
def build_dnn_model(input_shape):
    model = Sequential([
        Flatten(input_shape=input_shape),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])
    return model

# Federated Model Definition
def model_fn():
    keras_model = build_dnn_model(input_shape=(28, 28))
    keras_model.compile(optimizer='adam',
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])
    return tff.learning.from_keras_model(
        keras_model,
        input_spec=tf.TensorSpec(shape=[None, 28, 28, 1], dtype=tf.float32),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )

# Federated data preparation
def preprocess_for_federated_learning(dataset_name, load_data):
    (train_images, train_labels), (test_images, test_labels) = load_data()
    
    # Normalize images
    train_images, test_images = train_images / 255.0, test_images / 255.0
    
    # Split the data into clients
    num_clients = 10
    client_data_size = len(train_images) // num_clients
    federated_train_data = []
    
    for i in range(num_clients):
        start = i * client_data_size
        end = start + client_data_size
        federated_train_data.append(
            (train_images[start:end], train_labels[start:end])
        )
    
    return federated_train_data, (test_images, test_labels)

# Train Federated Learning model
def train_federated_model(federated_train_data):
    federated_model = model_fn()

    # Federated learning process
    iterative_process = tff.learning.build_federated_averaging_process(federated_model)
    state = iterative_process.initialize()

    # Train over several rounds
    num_rounds = 10
    for round_num in range(num_rounds):
        state, metrics = iterative_process.next(state, federated_train_data)
        print(f"Round {round_num+1}, Metrics: {metrics}")

    return state

# Evaluate model accuracy
def evaluate_model(federated_model, test_data):
    test_images, test_labels = test_data
    test_images = test_images / 255.0  # Normalize
    
    # Inference on the server side after aggregation
    keras_model = federated_model.create_keras_model()
    test_loss, test_acc = keras_model.evaluate(test_images, test_labels)
    print(f"Test Accuracy: {test_acc:.4f}")

# Logistic Regression Training function
def train_logistic_regression(train_images, train_labels, test_images, test_labels):
    # Flatten images for logistic regression
    train_images_flat = train_images.reshape(train_images.shape[0], -1)
    test_images_flat = test_images.reshape(test_images.shape[0], -1)
    
    # Scale the data
    scaler = StandardScaler()
    train_images_flat = scaler.fit_transform(train_images_flat)
    test_images_flat = scaler.transform(test_images_flat)
    
    # Train Logistic Regression model with increased max_iter and saga solver
    lr_model = LogisticRegression(max_iter=500, solver="saga")
    lr_model.fit(train_images_flat, train_labels)
    
    # Calculate accuracy
    train_accuracy = lr_model.score(train_images_flat, train_labels)
    test_predictions = lr_model.predict(test_images_flat)
    test_accuracy = accuracy_score(test_labels, test_predictions)
    return train_accuracy, test_accuracy

# Store results
results = []

# Parameters
epochs_dnn = 10  # Increased epochs

# Train and evaluate DNN and LR on all datasets
for dataset_name, load_data in datasets.items():
    print(f"\nTraining on {dataset_name} dataset")
    (train_images, train_labels), (test_images, test_labels) = load_data()

    # Normalize images for DNN
    train_images_dnn, test_images_dnn = train_images / 255.0, test_images / 255.0

    # Train DNN model
    dnn_model = build_dnn_model(input_shape=(28, 28))

    # Compile the model
    dnn_model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

    history = dnn_model.fit(train_images_dnn, train_labels, epochs=epochs_dnn, batch_size=32, verbose=0)
    train_acc_dnn = history.history['accuracy'][-1]  # Last training accuracy
    test_loss, test_acc_dnn = dnn_model.evaluate(test_images_dnn, test_labels, verbose=0)

    # Train Logistic Regression model
    train_acc_lr, test_acc_lr = train_logistic_regression(train_images, train_labels, test_images, test_labels)
    
    # Append results
    results.append({
        "Dataset": dataset_name,
        "Model": "DNN",
        "Training Accuracy": train_acc_dnn,
        "Test Accuracy": test_acc_dnn
    })
    results.append({
        "Dataset": dataset_name,
        "Model": "Logistic Regression",
        "Training Accuracy": train_acc_lr,
        "Test Accuracy": test_acc_lr
    })

    # Prepare Federated Learning data
    federated_train_data, test_data = preprocess_for_federated_learning(dataset_name, load_data)

    # Train Federated Learning model
    federated_model_state = train_federated_model(federated_train_data)

    # Evaluate the model
    evaluate_model(federated_model_state, test_data)

# Create DataFrame to display the results
df_results = pd.DataFrame(results)
print(df_results)

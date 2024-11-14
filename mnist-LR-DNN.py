import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

# Load MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess the data for both models
x_train_dnn = x_train / 255.0  # Normalize for DNN
x_test_dnn = x_test / 255.0

# Flatten data for Logistic Regression
x_train_lr = x_train.reshape(-1, 28*28) / 255.0
x_test_lr = x_test.reshape(-1, 28*28) / 255.0

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
lr_model = LogisticRegression(max_iter=100)
lr_model.fit(x_train_lr, y_train)

# Predict and evaluate Logistic Regression model
y_pred_lr = lr_model.predict(x_test_lr)
lr_accuracy = accuracy_score(y_test, y_pred_lr)
print(f"Logistic Regression Model Accuracy: {lr_accuracy:.4f}")

#Logistic Regression Model Accuracy: 0.9258
#DNN Model Accuracy: 0.9732
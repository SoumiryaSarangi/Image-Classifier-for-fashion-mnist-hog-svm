import numpy as np


def preprocess_data(X_train, X_test):
    # Normalize pixel values
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # Add channel dimension
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

    return X_train, X_test
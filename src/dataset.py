from tensorflow.keras.datasets import fashion_mnist


def load_dataset():
    """
    Load the Fashion MNIST dataset.
    """

    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

    return X_train, y_train, X_test, y_test
    
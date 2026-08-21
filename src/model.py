from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

import matplotlib.pyplot as plt
import joblib
import os

from src.config import (
    MODEL_PATH,
    CONFUSION_MATRIX_PATH,
    CLASSIFICATION_REPORT_PATH,
)


def train_model(X_train_hog, y_train):
    print("\nTraining SVM Model...")

    model = SVC(kernel="linear")

    model.fit(X_train_hog, y_train)

    print("Training Completed!")

    return model


def evaluate_model(model, X_train, y_train, X_test, y_test):

    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    train_accuracy = accuracy_score(y_train, train_predictions)
    test_accuracy = accuracy_score(y_test, test_predictions)

    print(f"\nTraining Accuracy : {train_accuracy*100:.2f}%")
    print(f"Testing Accuracy  : {test_accuracy*100:.2f}%")

    print("=" * 70)
    print("Classification Report")
    print("=" * 70)
    report = classification_report(
    y_test,
    test_predictions,
    digits=4
    )

    print(report)

    with open(CLASSIFICATION_REPORT_PATH, "w") as file:
        file.write(report)

    cm = confusion_matrix(y_test, test_predictions)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    fig, ax = plt.subplots(figsize=(8, 8))

    disp.plot(
    cmap="Blues",
    ax=ax,
    values_format="d"
    )

    plt.title("Confusion Matrix")

    plt.savefig(CONFUSION_MATRIX_PATH, dpi=300, bbox_inches="tight")

    plt.show()

    return train_predictions, test_predictions


def save_model(model):
    joblib.dump(model, MODEL_PATH)
    print("\nModel Saved Successfully!")

def load_model():
    model = joblib.load(MODEL_PATH)

    print("\nSaved model loaded successfully!")

    return model

def model_exists():

    return os.path.exists(MODEL_PATH)
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split

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


def analyze_confusion_matrix(cm, class_names, top_n=3):
    """Return (and print) the top_n most common misclassifications."""

    n = cm.shape[0]
    pairs = [
        (cm[i, j], class_names[i], class_names[j])
        for i in range(n)
        for j in range(n)
        if i != j
    ]

    pairs.sort(key=lambda x: x[0], reverse=True)

    lines = [
        "",
        "=" * 70,
        f"Top-{top_n} Most Confused Class Pairs",
        "=" * 70,
    ]

    for count, true_cls, pred_cls in pairs[:top_n]:
        lines.append(f"  {true_cls} confused as {pred_cls}: {count} times")

    analysis = "\n".join(lines)
    print(analysis)
    return analysis


def evaluate_model(model, X_train, y_train, X_test, y_test, class_names=None):

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

    if class_names is not None:
        analysis = analyze_confusion_matrix(cm, class_names)
        with open(CLASSIFICATION_REPORT_PATH, "a") as file:
            file.write(analysis + "\n")

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


def tune_model(X_train_hog, y_train, sample_size=8000, n_jobs=2):
    """Find best SVM hyperparams on a stratified subsample, then retrain on
    the full training set so the returned model has both optimal params and
    maximum data.

    Parameters
    ----------
    X_train_hog  : HOG feature matrix for the full training set.
    y_train      : Corresponding labels.
    sample_size  : Subset size for the grid search stage. Default 8 000.
                   Reduce further if you still hit memory limits.
    n_jobs       : Parallel workers for GridSearchCV. Default 2.
    """

    # --- stage 1: stratified subsample for cheap grid search ---------------
    if sample_size < len(y_train):
        X_sub, _, y_sub, _ = train_test_split(
            X_train_hog, y_train,
            train_size=sample_size,
            stratify=y_train,
            random_state=42,
        )
    else:
        X_sub, y_sub = X_train_hog, y_train

    print(f"\n[Stage 1] Running GridSearchCV on {len(y_sub):,} stratified "
          f"samples (full set: {len(y_train):,})...")

    param_grid = {
        "C":      [0.01, 0.1, 1, 10],
        "kernel": ["linear", "rbf"],
    }

    gs = GridSearchCV(
        SVC(),
        param_grid,
        cv=3,
        n_jobs=n_jobs,
        verbose=2,
    )

    gs.fit(X_sub, y_sub)

    best_params = gs.best_params_
    print(f"\nBest params found on {len(y_sub):,}-sample subset : {best_params}")
    print(f"Best CV Score (subset)                          : {gs.best_score_*100:.2f}%")

    # --- stage 2: retrain on the FULL training set with those params -------
    print(f"\n[Stage 2] Retraining on full {len(y_train):,}-sample training "
          f"set with params {best_params}...")

    final_model = SVC(**best_params)
    final_model.fit(X_train_hog, y_train)

    print("Retraining complete.")

    return final_model
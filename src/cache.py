import os
import joblib

from src.config import TRAIN_HOG_PATH, TEST_HOG_PATH


def hog_exists():
    return (
        os.path.exists(TRAIN_HOG_PATH)
        and
        os.path.exists(TEST_HOG_PATH)
    )


def save_hog_features(X_train_hog, X_test_hog):

    joblib.dump(X_train_hog, TRAIN_HOG_PATH)
    joblib.dump(X_test_hog, TEST_HOG_PATH)

    print("\nHOG Features Saved Successfully!")


def load_hog_features():

    X_train_hog = joblib.load(TRAIN_HOG_PATH)
    X_test_hog = joblib.load(TEST_HOG_PATH)

    print("\nSaved HOG Features Loaded!")

    return X_train_hog, X_test_hog
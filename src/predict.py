import cv2
import numpy as np
import joblib

from src.preprocessing import preprocess_data
from src.features import extract_hog_features


def load_model():
    return joblib.load("models/image_classifier.joblib")


def predict(model, X):
    return model.predict(X)


def predict_custom_image(image_path, model, class_names):

    image = cv2.imread(image_path)

    if image is None:
        print("Image not found!")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    resized = cv2.resize(gray, (28, 28))

    resized = resized.astype("float32") / 255.0

    resized = resized.reshape(1, 28, 28)

    processed, _ = preprocess_data(resized, resized)

    hog = extract_hog_features(processed)

    prediction = model.predict(hog)[0]

    print("\nPrediction :", class_names[prediction])

    cv2.imshow("Input Image", cv2.resize(image, (300, 300)))

    cv2.waitKey(0)

    cv2.destroyAllWindows()
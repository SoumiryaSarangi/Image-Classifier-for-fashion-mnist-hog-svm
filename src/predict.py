import cv2
import numpy as np
import matplotlib.pyplot as plt

from src.preprocessing import preprocess_data
from src.features import extract_hog_features
from src.model import load_model          # de-duplicated: use model.py's definition


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

    # Convert BGR -> RGB for correct colour rendering in matplotlib
    image_rgb = cv2.cvtColor(cv2.resize(image, (300, 300)), cv2.COLOR_BGR2RGB)

    plt.figure()
    plt.imshow(image_rgb)
    plt.title(f"Prediction: {class_names[prediction]}")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
from src.dataset import load_dataset
from src.visualize import show_sample_images, show_predictions
from src.preprocessing import preprocess_data
from src.features import extract_hog_features
from src.predict import predict_custom_image
from src.model import (
    train_model,
    evaluate_model,
    save_model,
    load_model,
    model_exists,
    tune_model,
)
from src.cache import (
    hog_exists,
    save_hog_features,
    load_hog_features
)

X_train, y_train, X_test, y_test = load_dataset()

class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

print("Dataset loaded successfully!\n")

print("Training Images :", X_train.shape)
print("Training Labels :", y_train.shape)

print()

print("Testing Images :", X_test.shape)
print("Testing Labels :", y_test.shape)

show_sample_images(X_train, y_train)

X_train, X_test = preprocess_data(X_train, X_test)

print("\nPreprocessing completed!\n")

print("Training Images :", X_train.shape)
print("Testing Images  :", X_test.shape)

if hog_exists():

    X_train_hog, X_test_hog = load_hog_features()

else:

    print("\nExtracting HOG Features...")

    X_train_hog = extract_hog_features(X_train)
    X_test_hog = extract_hog_features(X_test)

    save_hog_features(
        X_train_hog,
        X_test_hog
    )

print("\nFeature extraction completed!")

print("Training HOG Features :", X_train_hog.shape)
print("Testing HOG Features  :", X_test_hog.shape)

if model_exists():

    model = load_model()

else:

    model = train_model(X_train_hog, y_train)

    save_model(model)

while True:

    print("\nChoose an option:")
    print("1. Evaluate Model")
    print("2. Predict Custom Image")
    print("3. Tune Hyperparameters")
    print("4. Exit")

    choice = input("\nEnter your choice: ")

    if choice == "1":

        train_predictions, test_predictions = evaluate_model(
            model,
            X_train_hog,
            y_train,
            X_test_hog,
            y_test,
            class_names=class_names
        )

        show_predictions(
            X_test,
            y_test,
            test_predictions,
            class_names
        )

    elif choice == "2":

        image_path = input("\nEnter image path: ")

        predict_custom_image(
            image_path,
            model,
            class_names
        )

    elif choice == "3":

        tuned_model = tune_model(X_train_hog, y_train)

        save_choice = input("\nSave tuned model? (y/n): ").strip().lower()

        if save_choice == "y":
            save_model(tuned_model)
            model = tuned_model          # use tuned model for the rest of the session
            print("Tuned model is now active for this session.")
        else:
            print("Tuned model discarded. Existing model unchanged.")

    elif choice == "4":

        print("Exiting...")
        break

    else:

        print("Invalid choice, please try again.")
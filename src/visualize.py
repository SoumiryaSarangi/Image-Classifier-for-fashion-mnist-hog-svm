import matplotlib.pyplot as plt


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


def show_sample_images(images, labels):
    plt.figure(figsize=(15, 5))

    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[i], cmap="gray")
        plt.title(class_names[labels[i]])
        plt.axis("off")

    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt


def show_predictions(images, true_labels, predicted_labels, class_names):

    fig = plt.figure(figsize=(15,5))

    for i in range(10):

        plt.subplot(2,5,i+1)

        plt.imshow(images[i].reshape(28,28), cmap="gray")

        plt.title(
            f"T: {class_names[true_labels[i]]}\nP: {class_names[predicted_labels[i]]}",
            fontsize=9
        )

        plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        "outputs/predictions.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
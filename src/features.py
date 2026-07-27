import numpy as np
from skimage.feature import hog


def extract_hog_features(images):
    hog_features = []

    for image in images:
        features = hog(
            image,
            pixels_per_cell=(4, 4),
            cells_per_block=(2, 2),
            visualize=False,
            channel_axis=-1
        )

        hog_features.append(features)

    return np.array(hog_features)
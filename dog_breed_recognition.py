import numpy as np
import cv2


class DogResult:
    def __init__(self, breed, probability):
        self.breed = breed
        self.probability = probability

    def __repr__(self):
        return f'DogResult(breed={self.breed}, probability={self.probability})'


def read_image(img, size):
    image = cv2.imread(img, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (size, size))
    image = image / 255.0
    image = image.astype(np.float32)
    return image


def recogniseImg(img, model, id2breed):
    image = read_image(img, 224)
    image = np.expand_dims(image, axis=0)
    pred = model.predict(image)[0]
    label_idx = pred.argsort()[-3:][::-1]
    results = {}
    for i, idx in enumerate(label_idx):
        breed_name = id2breed[idx]
        probability = round(float(pred[idx]), 5) * 100
        dog_result = DogResult(breed_name, probability)
        results[i] = dog_result
    return results

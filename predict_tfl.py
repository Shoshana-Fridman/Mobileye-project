import numpy as np
from PIL import Image

from variables import CROPPING_SIZE

def get_cropping_edges(image, x, y, size):
    image_shape = image.shape
    left = max(0, x - size // 2)
    right = left + size
    if right > image_shape[0]-1:
        right = image_shape[0]-1
        left = right - size
    ceiling = max(y - size // 2, 0)
    bottom = ceiling + size
    if bottom > image_shape[1]-1:
        bottom = image_shape[1] - 1
        ceiling = bottom - size
    return left, right, ceiling, bottom


def crop_image(image, x, y, size):
    left, right, ceiling, bottom = get_cropping_edges(image, x, y, size)
    cropping_image = image[left:right, ceiling:bottom]
    return cropping_image


def is_tfl(model, image):
    crop_shape = (81, 81)
    test_image = image.reshape([-1] + list(crop_shape) + [3])
    predictions = model.predict(test_image)
    predicted_label = np.argmax(predictions, axis=-1)
    return predicted_label



def get_tfls(image, candidates, model):
    tfls = []
    for i, pixcel in enumerate(candidates):
        cropping_image = crop_image(image, candidates[i][1],candidates[i][0], CROPPING_SIZE)
        if is_tfl(model, cropping_image):
            tfls.append(pixcel)
    return tfls



def run(image_path, red_candidates, green_candidates, model):
    image = np.array(Image.open(image_path))
    red_tfls = get_tfls(image, red_candidates, model)
    green_tfls = get_tfls(image, green_candidates, model)
    return red_tfls, green_tfls





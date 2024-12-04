import cv2
import numpy as np

def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None or image.size == 0:
        raise FileNotFoundError(f"Gambar tidak ditemukan atau tidak bisa dibaca di {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def preprocess_image(image, target_size=(513, 513)):
    height, width = image.shape[:2]
    target_height, target_width = target_size

    aspect_ratio = width / height
    if aspect_ratio > 1:  # Lebar lebih besar dari tinggi
        new_width = target_width
        new_height = int(target_width / aspect_ratio)
    else:  # Tinggi lebih besar atau sama dengan lebar
        new_height = target_height
        new_width = int(target_height * aspect_ratio)

    image_resized = cv2.resize(image, (new_width, new_height))

    top = (target_height - new_height) // 2
    bottom = target_height - new_height - top
    left = (target_width - new_width) // 2
    right = target_width - new_width - left

    image_resized = cv2.copyMakeBorder(image_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    image_normalized = image_resized / 255.0
    return np.expand_dims(image_normalized, axis=0)


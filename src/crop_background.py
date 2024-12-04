import argparse
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils import load_image, preprocess_image 

def load_deeplabv3_model_from_tflite(model_path="models/1/1.tflite"):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def segment_image(interpreter, image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_index = input_details[0]['index']
    image_resized = cv2.resize(image, (513, 513))
    image_normalized = image_resized / 255.0  
    input_data = np.expand_dims(image_normalized, axis=0).astype(np.float32)

    interpreter.set_tensor(input_index, input_data)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    segmentation_mask = output_data[0]
    
    return segmentation_mask

def auto_crop(image, mask):
    indices = np.where(mask > 0)
    top_left = (min(indices[1]), min(indices[0]))
    bottom_right = (max(indices[1]), max(indices[0]))
    cropped_image = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    return cropped_image

def plot_image(image):
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def save_output(cropped_image, output_path="output/cropped_image.jpg"):
    cv2.imwrite(output_path, cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR))

def main(image_path):
    model_path = "models/1/1.tflite"  # Path ke model TFLite lokal
    interpreter = load_deeplabv3_model_from_tflite(model_path)

    image = load_image(image_path)  
    image = preprocess_image(image) 

    mask = segment_image(interpreter, image)
    cropped_image = auto_crop(image, mask)
    save_output(cropped_image)
    plot_image(cropped_image)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto crop background using DeepLabV3+ TFLite")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the image file")
    args = parser.parse_args()
    main(args.image_path)


import argparse
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils import load_image, preprocess_image  
import torchvision.models.segmentation as models

def load_deeplabv3_model(model_path="models/model.pth"):
    model = models.deeplabv3_resnet101(pretrained=False)
    
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    
    model.load_state_dict(state_dict)
    
    model.eval()  
    return model

def segment_image(model, image):
    input_image = preprocess_image(image, target_size=(513, 513))
    
    input_tensor = torch.from_numpy(input_image).float().unsqueeze(0)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_tensor = input_tensor.to(device)
    
    with torch.no_grad():  
        output = model(input_tensor)['out'][0]
    
    segmentation_mask = output.argmax(0).cpu().numpy()  
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
    model_path = "models/model.pth"  
    model = load_deeplabv3_model(model_path)

    image = load_image(image_path)  
    mask = segment_image(model, image)
    
    cropped_image = auto_crop(image, mask)
    save_output(cropped_image)
    plot_image(cropped_image)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto crop background using DeepLabV3 model in PyTorch")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the image file")
    args = parser.parse_args()
    main(args.image_path)


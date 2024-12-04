import kagglehub
import os
import shutil

model_dir = "./models/"

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

path = kagglehub.model_download("tensorflow/deeplabv3/tfLite/default")

new_model_path = os.path.join(model_dir, os.path.basename(path))

shutil.move(path, new_model_path)

print("Path ke file model:", new_model_path)



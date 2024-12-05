import torch
print(torch.__version__)  # Versi PyTorch yang terinstal
print(torch.cuda.is_available())  # Apakah CUDA tersedia
print(torch.cuda.get_device_name(0))  # Nama GPU pertama yang terdeteksi


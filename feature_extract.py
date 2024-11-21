import os
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

def read_images_to_tensor(directory):
    transform = transforms.ToTensor()
    image_tensors = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(directory, filename)
            image = Image.open(image_path)
            image_tensor = transform(image)
            image_tensors.append(image_tensor)
    return torch.stack(image_tensors)

def compute_vertical_gradient(images_tensor):
    kernel = torch.tensor([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    gradient = torch.nn.functional.conv2d(images_tensor, kernel, padding=1)
    gradient_image = gradient.abs().sum(dim=1, keepdim=True)
    return gradient, gradient_image

def compute_horizontal_gradient(images_tensor):
    kernel = torch.tensor([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    gradient = torch.nn.functional.conv2d(images_tensor, kernel, padding=1)
    gradient_image = gradient.abs().sum(dim=1, keepdim=True)
    return gradient, gradient_image

def save_images(tensor, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    to_pil = transforms.ToPILImage()
    for i, img_tensor in enumerate(tensor):
        img = to_pil(img_tensor)
        img.save(os.path.join(directory, f'image_{i}.png'))

# Example usage:
# images_tensor = read_images_to_tensor('/path/to/images')
# vertical_gradient, vertical_gradient_image = compute_vertical_gradient(images_tensor)
# horizontal_gradient, horizontal_gradient_image = compute_horizontal_gradient(images_tensor)
# save_images(vertical_gradient_image, '/path/to/save/vertical')
# save_images(horizontal_gradient_image, '/path/to/save/horizontal')
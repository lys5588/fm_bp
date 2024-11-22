import os
import torch
from torchvision import transforms
from PIL import Image
import numpy as np


def read_cifar10_to_tensor(directory):
    def unpickle(file):
        import pickle

        with open(file, "rb") as fo:
            dict = pickle.load(fo, encoding="bytes")
        return dict

    def extract_data_and_labels(batch):
        data = batch[b"data"]
        labels = batch[b"labels"]
        return data, labels

    all_images = []
    all_labels = []
    test_images = []
    test_labels = []
    for filename in os.listdir(directory):
        if filename.startswith("data_batch"):
            data = unpickle(os.path.join(directory, filename))
            images, labels = extract_data_and_labels(data)
            images = torch.tensor(images, dtype=torch.float32).reshape(-1, 3, 32, 32)
            labels = torch.tensor(labels)
            all_images.append(images)
            all_labels.append(labels)
        if filename.startswith("test_batch"):
            data = unpickle(os.path.join(directory, filename))
            images, labels = extract_data_and_labels(data)
            images = torch.tensor(images, dtype=torch.float32).reshape(-1, 3, 32, 32)
            labels = torch.tensor(labels)
            test_images.append(images)
            test_labels.append(labels)
    all_images = torch.cat(all_images, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    test_images = test_images[0]
    test_labels = test_labels[0]
    return all_images, all_labels, test_images, test_labels


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
    channels = images_tensor.shape[1]
    kernel = (
        torch.tensor([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=torch.float32)
        .unsqueeze(0)
        .unsqueeze(0)
    )
    kernel = kernel.repeat(1, channels, 1, 1)  # Repeat the kernel for each channel
    gradient = torch.nn.functional.conv2d(images_tensor, kernel, padding=1)
    gradient_image = gradient.abs().sum(dim=1, keepdim=True)
    return gradient, gradient_image


def compute_horizontal_gradient(images_tensor):
    channels = images_tensor.shape[1]
    kernel = (
        torch.tensor([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=torch.float32)
        .unsqueeze(0)
        .unsqueeze(0)
    )
    kernel = kernel.repeat(1, channels, 1, 1)  # Repeat the kernel for each channel
    gradient = torch.nn.functional.conv2d(images_tensor, kernel, padding=1)
    gradient_image = gradient.abs().sum(dim=1, keepdim=True)
    return gradient, gradient_image


def compute_gradiant(images_tensor):
    # 计算横向与纵向的梯度并合并
    vertical_gradient, vertical_gradient_image = compute_vertical_gradient(
        images_tensor
    )
    horizontal_gradient, horizontal_gradient_image = compute_horizontal_gradient(
        images_tensor
    )
    gradient = torch.cat((vertical_gradient, horizontal_gradient), dim=1)
    gradient_image = torch.cat(
        (vertical_gradient_image, horizontal_gradient_image), dim=1
    )
    return gradient, gradient_image


def save_images(tensor, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    to_pil = transforms.ToPILImage()
    for i, img_tensor in enumerate(tensor):
        img = to_pil(img_tensor)
        img.save(os.path.join(directory, f"image_{i}.png"))


def augment_data(input_images, input_labels):
    augmented_images = []
    augmented_labels = []

    transforms_list = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.RandomResizedCrop(size=(32, 32), scale=(0.8, 1.0)),
    ]

    for image, label in zip(input_images, input_labels):
        for transform in transforms_list:
            augmented_image = transform(image)
            augmented_images.append(augmented_image)
            augmented_labels.append(label)

    augmented_images_tensor = torch.stack(augmented_images)
    return augmented_images_tensor, augmented_labels


images = torch.randn(10, 1, 32, 32)
label = [i for i in range(10)]

augmented_images, augmented_labels = augment_data(images, label)
print(augmented_images.shape, augmented_labels)

# Example usage:
# images_tensor = read_images_to_tensor('/path/to/images')
# vertical_gradient, vertical_gradient_image = compute_vertical_gradient(images_tensor)
# horizontal_gradient, horizontal_gradient_image = compute_horizontal_gradient(images_tensor)
# save_images(vertical_gradient_image, '/path/to/save/vertical')
# save_images(horizontal_gradient_image, '/path/to/save/horizontal')

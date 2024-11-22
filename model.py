import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms
from tqdm import tqdm
import time
import os


class VGG16(nn.Module):
    def __init__(self, num_classes=1000):
        super(VGG16, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# Example usage:
# model = VGG16(num_classes=1000)
# print(model)
# # Create a random tensor with the shape (1, 3, 224, 224) to simulate a single image input
# input_tensor = torch.randn(1, 3, 224, 224)

# # Pass the tensor through the model
# output = model(input_tensor)

# # Print the output
# print(output.shape)


def vgg_train():
    epochs = 500  # 训练次数
    learning_rate = 1e-4  # 学习率
    batch_size = 16

    num_classes = 10

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = VGG16(num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()  # 交叉熵损失

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)  # Adam优化器

    total_start_time = time.time()
    for epoch in range(epochs):  # 迭代
        running_loss = 0.0
        epoch_start_time = time.time()
        for i, data in enumerate(
            tqdm(trainloader, desc=f"Epoch {epoch + 1}/{epochs}"), 0
        ):
            inputs, labels = data  # labels: [batch_size, 1]
            inputs, labels = inputs.to(device), labels.to(device)
            # 初始化梯度
            optimizer.zero_grad()
            # 前向传播
            outputs = net(inputs)  # outputs: [batch_size, 10]
            loss = criterion(outputs, labels)
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()

            # 打印loss
            running_loss += loss.item()
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        total_time = epoch_end_time - total_start_time
        # print(
        #     f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(trainloader):.5f}, Epoch Time: {epoch_time:.2f}s, Total Time: {total_time:.2f}s"
        # )

        # Save the model checkpoint
        if (epoch + 1) % 100 == 0:
            print(
                f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(trainloader):.5f}, Epoch Time: {epoch_time:.2f}s, Total Time: {total_time:.2f}s"
            )
            torch.save(
                net.state_dict(), f"./output/vgg16_cifar10_epoch_{epoch + 1}.pth"
            )
    print("Finished Training")


def vgg_test():
    batch_size = 16
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False
    )
    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = VGG16(num_classes=10).to(device)
    if not os.path.isfile("./output/vgg16_cifar10_epoch_400.pth"):
        raise FileNotFoundError("Model file not found. Please train the model first.")
    net.load_state_dict(torch.load("./output/vgg16_cifar10_epoch_400.pth"))
    net.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(
        f"Accuracy of the network on the 10000 test images: {100 * correct / total:.2f}%"
    )


vgg_test()

# vgg_train()

import torch
from torch.utils.data import DataLoader, TensorDataset

# Create a random tensor of shape [1000, 2, 32, 32]
data = torch.randn(1000, 2, 32, 32)

# Create a list of 1000 random integers between 0 and 9
labels = torch.randint(0, 10, (1000,))

# Create a TensorDataset and DataLoader
dataset = TensorDataset(data, labels)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Print the first two batches
for i, (batch_data, batch_labels) in enumerate(dataloader):
    if i >= 2:
        break
    print(f"Batch {i+1} data:\n", batch_data.shape)
    print(f"Batch {i+1} labels:\n", batch_labels)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import argparse
import os

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train MLP on XOR task')
parser.add_argument('--n_hidden', type=int, default=128, help='Number of hidden units')
parser.add_argument('--n_probes', type=int, default=1000, help='Number of random images chosen from test set for activation recording')
parser.add_argument('--seed', type=int, default=None, help='Random seed (optional)')
args = parser.parse_args()

n_hidden = args.n_hidden
seed = args.seed
n_probes = args.n_probes

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# set seed
if seed is not None:
    torch.manual_seed(seed)

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define MLP model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(784, n_hidden)
        self.fc2 = nn.Linear(n_hidden, 10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = x.view(-1, 784)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize model, loss, and optimizer
model = MLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Dictionary to store activations
activations = {}

# Hook to capture activations
def get_activation(name):
    def hook(module, input, output):
        activations[name] = output.detach().cpu()
    return hook


# Train the model
epochs = 10
for epoch in range(epochs):
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# Evaluate the model
correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = correct / total
print(f"Test Accuracy: {test_accuracy:.4f}")

# Save model weights
activations["model_weights"] = {name: param.detach().cpu() for name, param in model.named_parameters()}

# Save accuracy
activations["test_accuracy"] = test_accuracy

# Get activations randomly chosen test images
n_samples = n_probes
random_indices = np.random.choice(len(test_dataset), n_samples, replace=False)
random_subset = Subset(test_dataset, random_indices)
random_loader = DataLoader(random_subset, batch_size=n_samples, shuffle=False)

# Get the random images and their labels
random_images, random_labels = next(iter(random_loader))
random_images, random_labels = random_images.to(device), random_labels.to(device)

# Register hook on the layers
hook_handle_relu = model.relu.register_forward_hook(get_activation('relu'))
hook_handle_2 = model.fc2.register_forward_hook(get_activation('fc2'))
hook_handle_1 = model.fc1.register_forward_hook(get_activation('fc1'))

# Forward pass to capture activations
with torch.no_grad():
    _ = model(random_images)

# Store activations for later use
activations_dict = {
    'fc1': activations['fc1'].clone(),
    'relu': activations['relu'].clone(),
    'fc2': activations['fc2'].clone(),
    'images': random_images.cpu(),
    'labels': random_labels.cpu(),
    'accuracy': test_accuracy,
    'model_weights': activations['model_weights']
}

activations_dict["seed"] = seed
activations_dict["n_samples"] = n_samples
activations_dict["n_hidden"] = n_hidden
activations_dict["probe_images"] = random_images.cpu()
activations_dict["probe_labels"] = random_labels.cpu()

# Remove hooks
hook_handle_relu.remove()
hook_handle_2.remove()
hook_handle_1.remove()

# Check if directory exists, if not create it
if not os.path.exists("trained_activations"):
    os.makedirs("trained_activations")

torch.save(activations_dict, f"trained_activations/mlp_MNIST_activations_seed{str(seed)}_nhidden{n_hidden}_nprobes{n_samples}.pt")

print(f"\nCaptured activations for {n_samples} random test images:")
print(f"  fc1 activations shape: {activations_dict['fc1'].shape}")
print(f"  relu activations shape: {activations_dict['relu'].shape}")
print(f"  fc2 activations shape: {activations_dict['fc2'].shape}")
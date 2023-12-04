import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F

# Define the Conv2dCNN model
class Conv2dCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(Conv2dCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Continual Learning                    
def train_model(model, train_loader, optimizer, criterion, device):
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)

        # Flatten the outputs and labels to have the same batch size
        outputs = outputs.view(-1, outputs.size(-1))
        labels = labels.view(-1)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Test model
def test_model(model, test_loader, dataset_name, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    print(f'Test Accuracy on {dataset_name}: {accuracy * 100:.2f}%')

# Elastic Weight Consolidation (EWC) for Stability
class EWCRegularizer:
    def __init__(self, model, dataloader, criterion, fisher_multiplier):
        self.model = model
        self.dataloader = dataloader
        self.criterion = criterion
        self.fisher_multiplier = fisher_multiplier
        self.fisher_dict = self.calculate_fisher()
        self.start_params = {name: param.clone().detach() for name, param in model.named_parameters()}

    def calculate_fisher(self):
        model_params = [(name, param) for name, param in self.model.named_parameters() if param.requires_grad]
        fisher_dict = {}

        for idx, (name, param) in enumerate(model_params):
            fisher_dict[name] = torch.zeros_like(param.data)

        self.model.eval()
        for inputs, labels in self.dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels.long())

            self.model.zero_grad()
            loss.backward()

            for name, param in model_params:
                fisher_dict[name] += param.grad.data ** 2 / len(self.dataloader)

        return fisher_dict

    def penalty(self):
        model_params = [param for name, param in self.model.named_parameters() if param.requires_grad]
        loss = 0
        for name, param in self.model.named_parameters():
            fisher_info = self.fisher_dict[name]
            loss += (fisher_info * (param - self.start_params[name]) ** 2).sum()

        return 0.5 * self.fisher_multiplier * loss

# Synaptic Intelligence (SI) for Plasticity
# Synaptic Intelligence (SI) for Plasticity
class SIRegularizer:
    def __init__(self, model, dataloader, criterion, importance_multiplier, num_classes):
        self.model = model
        self.dataloader = dataloader
        self.criterion = criterion
        self.importance_multiplier = importance_multiplier
        self.num_classes = num_classes
        self.importance_dict = self.calculate_importance()
        self.start_params = {name: param.clone().detach() for name, param in model.named_parameters()}

    def calculate_importance(self):
        model_params = [(name, param) for name, param in self.model.named_parameters() if param.requires_grad]
        importance_dict = {}
        for name, param in model_params:
            importance_dict[name] = torch.zeros_like(param.data)

        self.model.eval()
        for inputs, labels in self.dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device).to(torch.long)  # Convert labels to integers
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            self.model.zero_grad()
            loss.backward()

            for name, param in model_params:
                importance_dict[name] += torch.abs(param.grad.data) / len(self.dataloader)

        return importance_dict


    def penalty(self):
        model_params = [param for name, param in self.model.named_parameters() if param.requires_grad]
        loss = 0
        for name, param in self.model.named_parameters():
            importance_info = self.importance_dict[name]
            loss += (importance_info * (param - self.start_params[name]) ** 2).sum()

        return 0.5 * self.importance_multiplier * loss


# Training parameters
device = torch.device( "cuda")
epochs = 100
learning_rate = 0.01

# Define transformations
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load datasets
cifar10_train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
cifar10_test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

cifar100_train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
cifar100_test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

caltech101_train_dataset = datasets.ImageFolder(root='./data/101_ObjectCategories', transform=transform)
caltech101_test_dataset = datasets.ImageFolder(root='./data/101_ObjectCategories', transform=transform)

# Create data loaders
cifar10_train_loader = DataLoader(cifar10_train_dataset, batch_size=64, shuffle=True)
cifar10_test_loader = DataLoader(cifar10_test_dataset, batch_size=64, shuffle=False)

cifar100_train_loader = DataLoader(cifar100_train_dataset, batch_size=32, shuffle=True)
cifar100_test_loader = DataLoader(cifar100_test_dataset, batch_size=32, shuffle=False)

caltech101_train_loader = DataLoader(caltech101_train_dataset, batch_size=32, shuffle=True)
caltech101_test_loader = DataLoader(caltech101_test_dataset, batch_size=32, shuffle=False)


# Define the model for CIFAR10
model_cifar10 = Conv2dCNN(num_classes=10).to(device)
optimizer_cifar10 = optim.SGD(model_cifar10.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# Continual Learning on CIFAR10
for epoch in range(epochs):
    train_model(model_cifar10, cifar10_train_loader, optimizer_cifar10, criterion, device)
test_model(model_cifar10, cifar10_test_loader, 'CIFAR10', device)

# Save model state for future tasks
#torch.save(model_cifar10.state_dict(), 'cifar10_model.pth')

# Elastic Weight Consolidation (EWC) for Stability on CIFAR100
model_cifar100 = Conv2dCNN(num_classes=100).to(device)

ewc_regularizer_cifar100 = EWCRegularizer(model_cifar100, cifar100_train_loader, criterion, fisher_multiplier=1e3)
optimizer_cifar100 = optim.SGD(model_cifar100.parameters(), lr=learning_rate)

for epoch in range(epochs):
    train_model(model_cifar100, cifar100_train_loader, optimizer_cifar100, criterion, device)
    ewc_regularizer_cifar100.set_model(model_cifar100)  # Update model for EWC
test_model(model_cifar100, cifar100_test_loader, 'CIFAR100', device)

# Save model state for future tasks
# torch.save(model_cifar100.state_dict(), 'cifar100_model.pth')

# Synaptic Intelligence (SI) for Plasticity on Caltech101
model_caltech101 = Conv2dCNN(num_classes=101).to(device)

si_regularizer_caltech101 = SIRegularizer(model_caltech101, caltech101_train_loader, criterion, importance_multiplier=1e3, num_classes=101)
optimizer_caltech101 = optim.SGD(model_caltech101.parameters(), lr=learning_rate)

for epoch in range(epochs):
    train_model(model_caltech101, caltech101_train_loader, optimizer_caltech101, criterion, device)
    si_regularizer_caltech101.set_model(model_caltech101)  # Update model for SI
test_model(model_caltech101, caltech101_test_loader, 'Caltech101', device)

# Save model state for future tasks
torch.save(model_caltech101.state_dict(), 'caltech101_model.pth')

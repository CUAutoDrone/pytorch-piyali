# import statements
import numpy as np
import torch
from torch import nn
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt


# defining FFNN class; creating neural network setup
class FFNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer1 = nn.Linear(784, 196)
        self.linear_layer2 = nn.Linear(196, 49)
        self.linear_layer3 = nn.Linear(49, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        # using activation functions
        x = F.relu(self.linear_layer1(x))
        x = F.relu(self.linear_layer2(x))
        x = F.relu(self.linear_layer3(x))
        return x


# creating the FFNN
model = FFNN()

# training the model
train_dataset = torchvision.datasets.MNIST(
    "files/",
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor(),
    target_transform=torchvision.transforms.Compose(
        [lambda x: torch.LongTensor([x]), lambda x: F.one_hot(x, 10)]
    ),
)

# testing dataset
test_dataset = torchvision.datasets.MNIST(
    "files/",
    train=False,
    download=False,
    transform=torchvision.transforms.ToTensor(),
    target_transform=torchvision.transforms.Compose(
        [lambda x: torch.LongTensor([x]), lambda x: F.one_hot(x, 10)]
    ),
)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=60, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=60, shuffle=True)

for input, label in train_loader:
    model(input)
    break


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
num_epochs = 10

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_epoch(model, optimizer, dataloader):
    model.train()  # You have to tell your model to go into "train" mode
    train_losses = []
    for input, labels in dataloader:
        input, labels = input.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, labels.squeeze(1).float())
        train_losses.append(loss.item())
        loss.backward()
        optimizer.step()
    return train_losses  # Typically you want to keep a list of all the batch losses


def test_epoch(model, dataloader):
    model.eval()
    test_losses = []

    with torch.no_grad():
        for input, labels in dataloader:
            input, labels = input.to(DEVICE), labels.to(DEVICE)
            output = model(input)
            loss = criterion(output, labels.squeeze(1).float())
            test_losses.append(loss.item())

    return test_losses


overall_train_losses = []
overall_test_losses = []

for i in range(1, (num_epochs + 1)):
    train_losses = train_epoch(model, optimizer, train_loader)
    overall_train_losses.append(np.average(train_losses))
    print(i)
    test_losses = test_epoch(model, test_loader)
    overall_test_losses.append(np.average(test_losses))
    print(i)

print(overall_train_losses)
print(overall_test_losses)

# plotting losses
plt.plot(list(range(1, num_epochs + 1)), overall_train_losses, label="Training Loss")
plt.plot(list(range(1, num_epochs + 1)), overall_test_losses, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

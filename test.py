import numpy as np
import torch
from torch import nn
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt


class FFNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer1 = nn.Linear(784, 392)
        self.linear_layer2 = nn.Linear(392, 196)
        self.linear_layer3 = nn.Linear(196, 98)
        self.linear_layer4 = nn.Linear(98, 49)
        self.linear_layer5 = nn.Linear(49, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.linear_layer1(x))
        x = F.relu(self.linear_layer2(x))
        x = F.relu(self.linear_layer3(x))
        x = F.relu(self.linear_layer4(x))
        x = F.relu(self.linear_layer5(x))

        return x


model = FFNN()


train_dataset = torchvision.datasets.MNIST(
    "files/",
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor(),
    target_transform=torchvision.transforms.Compose(
        [lambda x: torch.LongTensor([x]), lambda x: F.one_hot(x, 10)]
    ),
)

test_dataset = torchvision.datasets.MNIST(
    "files/",
    train=False,
    download=False,
    transform=torchvision.transforms.ToTensor(),
    target_transform=torchvision.transforms.Compose(
        [lambda x: torch.LongTensor([x]), lambda x: F.one_hot(x, 10)]
    ),
)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=False)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

DEVICE = torch.device("cpu")
training_loss = []


def train_epoch(model, optimizer, dataloader, criterion):
    model.train()  # You have to tell your model to go into "train" mode
    losses = []
    for input, labels in dataloader:
        input, labels = input.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, labels)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    return losses  # Typically you want to keep a list of all the batch losses

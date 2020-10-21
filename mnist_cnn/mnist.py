import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer1 = self.cnn_layer(1,16)
        self.layer2 = self.cnn_layer(16,32)
        self.layer3 = self.cnn_layer(32,64)
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.linear1 = self.linear_layer(64,16)
        self.linear2 = self.linear_layer(16,10,dropout_rate=0)

    def cnn_layer(self, in_f, out_f, kernel_size=(3,3), stride=(2,2), padding=(1,1), dropout_rate=0.2):
        return nn.Sequential(
            nn.Conv2d(in_f, out_f, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_f),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate)
        )

    def linear_layer(self, in_f, out_f, dropout_rate = 0.5):
        return nn.Sequential(
            nn.Linear(in_f, out_f),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate)
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(-1,64)
        out = self.linear1(out)
        out = self.linear2(out)
        return out

# device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model().double().to(device)
optimizer = optim.Adam(model.parameters(), lr = 0.00025, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3, verbose=True)
cross_entropy = nn.CrossEntropyLoss()
transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
train_loader, test_loader = DataLoader(MNIST('./mnist', transform=transform, train=True, download=True), batch_size=60, shuffle=True, num_workers=4),\
    DataLoader(MNIST('./mnist', transform=transform, train=False, download=True), batch_size=60, shuffle=True, num_workers=4)
max_epochs, train_losses, test_losses = 50, [], []

for epoch in range(max_epochs):
    model.train()
    train_loss, test_loss = [], []
    for i_batch, (data, labels) in enumerate(train_loader):
        data, labels = data.double().to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = cross_entropy(output, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
        optimizer.step()
        train_loss.append(loss.item())
    train_loss = np.mean(train_loss)
    scheduler.step(train_loss)
    train_losses.append(train_loss)
    model.eval()
    with torch.no_grad():
        for i_batch, (data, labels) in enumerate(test_loader):
            data, labels = data.double().to(device), labels.to(device)
            output = model(data)
            loss = cross_entropy(output, labels)
            test_loss.append(loss.item())
    test_loss = np.mean(test_loss)
    test_losses.append(test_loss)
    print(f'Epoch [{epoch+1}/{max_epochs}] -> Train loss: {train_loss:.4f}, Test loss: {test_loss:.4f}')

plt.figure()
plt.plot([i for i in range(len(train_losses))], train_losses)
plt.title('Train Loss (MSE) vs Epochs')
# plt.yscale('log')
plt.savefig('train_loss.png')

plt.figure()
plt.plot([i for i in range(len(test_losses))], test_losses)
plt.title('Test Loss (MSE) vs Epochs')
# plt.yscale('log')
plt.savefig('test_loss.png')
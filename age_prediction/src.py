import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import torchvision
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize
import pandas as pd

def process_csv():
    with open('./data/age_gender.csv', 'r') as file:
        ages, ethicities, genders, data = [], [], [], []
        mean_img, count = np.zeros((48*48,)), 0
        file.readline() # skip header
        for line in file:
            age, ethnicity, gender, _, d = line.strip().split(',')
            ages.append(int(age))
            ethicities.append(int(ethnicity))
            genders.append(int(gender))
            img = (np.asarray([int(x) for x in d.split(' ')]).astype(np.float32) / 255.)
            mean_img += img
            count += 1
            data.append(img)

        ages, ethicities, genders, data = np.asarray(ages), np.asarray(ethicities), np.asarray(genders), np.asarray(data)
        ages = (ages - np.amin(ages))/np.amax(ages)
        data -= (mean_img / float(count))[np.newaxis,:]

        with open('./data/ages.txt', 'w') as f:
            np.savetxt(f, ages)

        with open('./data/ethicities.txt', 'w') as f:
            np.savetxt(f, ethicities)

        with open('./data/genders.txt', 'w') as f:
            np.savetxt(f, genders)

        with open('./data/pixels.txt', 'w') as f:
            np.savetxt(f, data)

def get_dataloaders(datapath, labelpath, split_ratio = 0.8):
    with open(labelpath, 'r') as file:
        labels = np.loadtxt(file)

    with open(datapath, 'r') as file:
        data = np.loadtxt(file)

    class_elements = [np.argwhere(labels == label) for label in np.unique(labels)]
    test, train = [], []

    for s in class_elements:
        split = int(len(s) * split_ratio)
        train_split, test_split = [i[0] for i in s[:split]], [i[0] for i in s[split:]]
        train.extend(train_split)
        test.extend(test_split)
    return DataLoader(TensorDataset(torch.from_numpy(data[train,:]).view(-1,48,48), torch.LongTensor(labels[train])), batch_size=60, shuffle=True),\
        DataLoader(TensorDataset(torch.from_numpy(data[test,:]).view(-1,48,48), torch.LongTensor(labels[test])), batch_size=60, shuffle=True)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer1 = self.cnn_layer(1,16) # 24
        self.layer2 = self.cnn_layer(16,32) # 12
        self.layer3 = self.cnn_layer(32,64) # 6
        self.layer4 = self.cnn_layer(64,128) # 3
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.linear1 = self.linear_layer(128,32)
        self.linear2 = nn.Linear(32, 1)

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
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(-1,128)
        out = self.linear1(out)
        out = self.linear2(out)
        return out

process_csv()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model().double().to(device)
optimizer = optim.Adam(model.parameters(), lr = 0.00001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3, verbose=True)
mse_loss = nn.MSELoss()
train_loader, test_loader = get_dataloaders('./data/pixels.txt','./data/ages.txt')
max_epochs, train_losses, test_losses = 50, [], []

for epoch in range(max_epochs):
    model.train()
    train_loss, test_loss = [], []
    for i_batch, (data, labels) in enumerate(train_loader):
        data, labels = data.view(-1,1,48,48).double().to(device), labels.view(-1,1).double().to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = mse_loss(output, labels)
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
            data, labels = data.view(-1,1,48,48).double().to(device), labels.view(-1,1).double().to(device)
            output = model(data)
            loss = mse_loss(output, labels)
            test_loss.append(loss.item())
    test_loss = np.mean(test_loss)
    test_losses.append(test_loss)
    print(f'Epoch [{epoch+1}/{max_epochs}] -> Train loss: {train_loss:.6f}, Test loss: {test_loss:.6f}')

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

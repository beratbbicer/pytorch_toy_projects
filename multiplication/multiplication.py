import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, TensorDataset

def get_dataloaders(halfsize=30000, range=1000):
    data1 = torch.rand(halfsize, 2)*range
    target1 = (data1[:,0]*data1[:,1]).view(-1,1)
    data2 = torch.rand(halfsize, 2)*-1*range
    target2 = (data2[:,0]*data2[:,1]).view(-1,1)
    data, target = torch.cat((data1, data2),dim=0), torch.cat((target1, target2),dim=0)
    dmin, tmin = data.min(1, keepdim=True)[0], target.min(1, keepdim=True)[0]
    dmax, tmax = data.max(1, keepdim=True)[0], target.max(1, keepdim=True)[0]
    data, target = (data - dmin)/dmax, (target - tmin)/tmax
    train_data, test_data = data[:50000,:], data[50000:,:]
    train_target, test_target = target[:50000,:], target[50000:,:]
    return DataLoader(TensorDataset(train_data, train_target), batch_size=60, shuffle=True),\
        DataLoader(TensorDataset(test_data, test_target), batch_size=60, shuffle=True)
    
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(2, 64),
            nn.BatchNorm1d(64),
            nn.Tanh(),
            nn.Dropout(p=0.2),
            nn.Linear(64,64),
            nn.BatchNorm1d(64),
            nn.Tanh(),
            nn.Linear(64,8),
            nn.BatchNorm1d(8),
            nn.Tanh(),
            nn.Dropout(p=0.2),
            nn.Linear(8,1)
        )

    def forward(self, x):
        return self.layer(x)

max_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mse_loss = nn.MSELoss()
model = Model().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001, weight_decay=1e-5)
train_loader, test_loader = get_dataloaders()
train_losses, test_losses = [],[]
idx = 0
for epoch in range(max_epochs):
    print(f'{idx}')
    model.train()
    train_loss, test_loss = [], []
    for i_batch, (data, targets) in enumerate(train_loader):
        data, targets = data.to(device), targets.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = mse_loss(output, targets)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
        optimizer.step()
        train_loss.append(loss.item())
    train_losses.append(np.mean(train_loss))
    model.eval()
    with torch.no_grad():
        for i_batch, (data, targets) in enumerate(test_loader):
            data, targets = data.to(device), targets.to(device)
            output = model(data)
            loss = mse_loss(output, targets)
            test_loss.append(loss.item())
    test_losses.append(np.mean(test_loss))
    idx += 1

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
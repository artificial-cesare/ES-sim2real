"""
Run as:
    mpirun -n (n_workers) python example_es.py

Important: if using a shared GPU on the same node, you should also run:
    # nvidia-cuda-mps-control -d

"""

import time

import torch
import torch as th
import torchvision
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0")
#device = torch.device("cpu")


n_epochs = 15
batch_size_train = 256
batch_size_test = 1000
log_interval = 10

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('mnist/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('mnist/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_test, shuffle=True)


network = nn.Sequential(nn.Conv2d(1, 16, kernel_size=5),
                        nn.MaxPool2d(2),
                        nn.ReLU(),
                        nn.Conv2d(16, 32, kernel_size=5),
                        nn.MaxPool2d(2),
                        nn.ReLU(),
                        nn.Flatten(),
                        nn.Linear(512, 64),
                        nn.ReLU(),
                        nn.Linear(64, 10),
                        nn.LogSoftmax()
                       ).to(device)

optimizer = optim.Adam(network.parameters(), lr=0.001)

def test():
  network.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      data, target = data.to(device), target.to(device)
      output = network(data)
      test_loss += F.nll_loss(output, target, size_average=False).item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  test_loss /= len(test_loader.dataset)
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))

def train_full_gradient():
    for epoch in range(1, n_epochs + 1):
      network.train()
      for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
          print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))
    #torch.save(network.state_dict(), 'model.pth')
    #torch.save(optimizer.state_dict(), 'optimizer.pth')


#train_full_gradient()
#test()  ## Baseline: 98.43%





from esmpi import ESMPI

optimizer = ESMPI(network, learning_rate=0.003, sigma=0.02, population_size=16*12, device=device)

def train_es():
    last_t = time.time()
    for epoch in range(1, n_epochs + 1):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            ###
            def eval_fn(model):
                output = model(data)
                loss = F.nll_loss(output, target).cpu().item()
                return loss
            loss = optimizer.step(eval_fn)
            ###

            if batch_idx % log_interval == 0  and  optimizer.is_master:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                  epoch, batch_idx * len(data), len(train_loader.dataset),
                  100. * batch_idx / len(train_loader), loss.item()))
                print('\t time: ', time.time()-last_t)
                last_t = time.time()

        if optimizer.is_master:
            print('EVAL at the end of each epoch: ')
            test()  ## Baseline: 98.43%

    #torch.save(network.state_dict(), 'model.pth')
    #torch.save(optimizer.state_dict(), 'optimizer.pth')


train_es()

if optimizer.is_master:
    test()  ## Baseline: 98.43%



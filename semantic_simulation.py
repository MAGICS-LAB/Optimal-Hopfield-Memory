import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
import torch.utils.data as data_utils
from torchvision import datasets, transforms


class FeatureMap(nn.Module):
    def __init__(self, inp_dim, hid_dim):
        super().__init__()

        self.phi = nn.Linear(inp_dim, hid_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.phi.weight.data, nonlinearity="relu")

    def forward(self, x):
        # return self.phi(x)
        return self.phi(x)

        out = torch.matmul(self.phi.weight.T, x.T).T
        return out


def loss_fn(x, tau=1):

    inner = -0.5*(F.pdist(x, p=2)**2 -2)
    loss = torch.log(torch.exp(inner).sum()).sum()
    return loss

N = 1

def get_classwise_data():

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    testset = datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    data = {}

    for x, y in testset:
        if y not in data.keys():
            data[y] = [x]
        else:
            data[y].append(x)

    return data, classes


data, classes = get_classwise_data()

cat = data[3][0:N]
dog = data[5][0:N]

ship = data[8][0:N]
truck = data[9][0:N]

memory = cat + dog + ship + truck

memory = torch.stack(memory, dim=0)

memory = memory.view(4, -1)

print(memory.size())

w = FeatureMap( memory.size(-1), 2).cuda()
memory = memory.cuda()
opt = torch.optim.Adam(w.parameters(), lr=0.01)

for i in range(50):

    opt.zero_grad()
    
    out = w(memory)
    loss = loss_fn(out)
    loss.backward()
    opt.step()

# visualize

x = w(memory)

count = 0

cat = x[count:N].numpy()
dogs = x[count+N:count+2*N].numpy()
ship = x[count+2*N:count+3*N].numpy()
truck = x[count+3*N:count+4*N].numpy()

import matplotlib.pyplot as plt

plt.scatter( cat[:, 0], cat[:, 1], label='cat')
plt.scatter( dog[:, 0], dog[:, 1], label='dog')
plt.scatter( ship[:, 0], ship[:, 1], label='ship')
plt.scatter( truck[:, 0], truck[:, 1], label='truck')

plt.legend()
plt.show()
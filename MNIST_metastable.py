# %%
import sys
sys.path.append("..")
from utils import HopfieldNet, Flatten # , normmax_bisect
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from utils import entmax # , SparseMAP_exactly_k
from collections import Counter
import numpy as np

from prettytable import PrettyTable   
from feature_map import FeatureMap, train_separation

import argparse

torch.set_num_threads(5)

parser = argparse.ArgumentParser(description="Synthetic Metastable State")
parser.add_argument("--D", type=int, default=784, help="dimension of memory pattern")
parser.add_argument("--N", type=int, default=5, help="number of memories")
parser.add_argument("--D_phi", type=int, default=200, help="dimension of feature space")

parser.add_argument("--iteration", type=int, default=1, help="iteration of separation training")
parser.add_argument("--lr", type=float, default=0.1, help="lr of separation training")
parser.add_argument("--tau", type=float, default=0.1, help="separation loss temperature")
parser.add_argument("--batch_size", type=int, default=16, help="separation training batch size")

args = parser.parse_args()


# %%
torch.random.manual_seed(42)

# Define the transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.5,0.5),
    Flatten()  # Normalize to [-1, 1]
])

# Load the MNIST dataset
mnist_dataset = datasets.MNIST(root='../datasets', train=True, download=True, transform=transform)
mnist_dataset_test = datasets.MNIST(root='../datasets', train=False, download=True, transform=transform)

# # Create a DataLoader to iterate over the dataset
# data_loader = torch.utils.data.DataLoader(mnist_dataset, batch_size=len(mnist_dataset), shuffle=True)
# # Create a DataLoader to iterate over the dataset
# data_loader_test = torch.utils.data.DataLoader(mnist_dataset_test, batch_size=len(mnist_dataset_test), shuffle=True)


# Create a DataLoader to iterate over the dataset
data_loader = torch.utils.data.DataLoader(mnist_dataset, batch_size=args.batch_size, shuffle=True)
# Create a DataLoader to iterate over the dataset
data_loader_test = torch.utils.data.DataLoader(mnist_dataset_test, batch_size=args.batch_size, shuffle=True)

X_train = []
X_test = []

for data in data_loader:
    img, labels_train = data
    X_train.append(img)

for data in data_loader_test:
    img, labels_test = data
    X_test.append(img)
    
X_train = torch.cat(X_train, dim=0)
X_test = torch.cat(X_test, dim=0)

# for data in data_loader:
#     X_train, labels_train = data

# for data in data_loader_test:
#     X_test, labels_test = data



def cccp(X, Q, alpha, beta, num_iters, k = None, normmax = False):

    results = []
    for i in range(Q.size(-1)):
        Xi = Q[:, i].unsqueeze(-1) # query
        for _ in range(num_iters):
            P = entmax(X @ Xi *beta, alpha=alpha, dim=0)
            Xi = X.T @ P

        results.append(P)
    results = torch.cat( results, dim=-1)
    return results

def kernelized_cccp(X, raw_memory, Q, alpha, beta, num_iters, w):

    # # calculate Phi(memory)
    # memories_feat = []
    # for i in range(X.size(0)):
    #     memories_feat.append(X[i])

    # memories_feat = torch.cat(memories_feat, dim=0)

    results = []
    for i in range(Q.size(-1)):
        Xi = Q[:, i].unsqueeze(-1) # query
        for _ in range(num_iters):
            P = entmax(X @ w(Xi.T).T * beta, alpha=alpha, dim=0)
            Xi = raw_memory.T @ P

        results.append(P)
    results = torch.cat( results, dim=-1)
    return results


num_iters = 5
eps = 1e-2
# device = torch.device("cuda:" + "0")

# Specify the Column Names while initializing the Table 

device = torch.device("cuda")
X_train = X_train.to(device)
X_test = X_test.to(device)
n_samples = X_test.shape[0]
N = X_train.shape[0]
ctrs_total = []

print("running")


def run_cccp(X_train, X_test, num_iters):
    tab = PrettyTable(["K", "beta", "alpha", "metastable size"]) 

    ctrs_total = []
    n_samples = X_test.shape[0]
    N = X_train.shape[0]

    for beta in [0.1]:
        ctrs = []
        for alpha in [1]:
            with torch.no_grad():
                P = cccp(X_train, X_test.T, alpha, beta, num_iters)

            eps_ = eps if alpha == 1 else 0
            sizes = (P > eps_).sum(dim=0)

            ctr = Counter(sizes.tolist())
            ctrs.append(ctr)

        ctrs_total.append(ctrs)

    for k in range(1, 11):

        for i, beta in enumerate([0.1]):
            for j, alpha in enumerate([1]):
                score = round(ctrs_total[i][j][k]/n_samples * 100, 2)
                tab.add_row([k, beta, alpha, score]) 

    print("CCCP")
    print(tab)


def run_kernel_cccp(X_train, X_test, num_iters, w):
    tab = PrettyTable(["K", "beta", "alpha", "metastable size"]) 

    ctrs_total = []
    n_samples = X_test.shape[0]
    N = X_train.shape[0]
    memories_feat = []
    for i in range(X_train.size(0)):
        memories_feat.append(w(X_train[i].unsqueeze(0)))
    
    memories_feat = torch.cat(memories_feat, dim=0)

    for beta in [0.1]:
        ctrs = []
        for alpha in [1]:
            
            with torch.no_grad():
                P = kernelized_cccp(memories_feat, X_train, X_test.T, alpha, beta, num_iters, w)

            eps_ = eps if alpha == 1 else 0
            sizes = (P > eps_).sum(dim=0)

            ctr = Counter(sizes.tolist())
            ctrs.append(ctr)

        ctrs_total.append(ctrs)

    for k in range(1, 11):

        for i, beta in enumerate([0.1]):
            for j, alpha in enumerate([1]):
                score = round(ctrs_total[i][j][k]/n_samples * 100, 2)
                tab.add_row([k, beta, alpha, score]) 

    print("kernelized CCCP")
    print(tab)


run_cccp(X_train, X_test, num_iters)
print("training")
w = train_separation(data_loader, args.D, args.D_phi, args.iteration, args.lr, args.tau)
run_kernel_cccp(X_train, X_test, num_iters, w)

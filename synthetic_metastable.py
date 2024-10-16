# %%
import sys

sys.path.append("..")
# from utils import HopfieldNet, Flatten, normmax_bisect
import torch

# from torchvision import datasets, transforms
# import matplotlib.pyplot as plt
from utils import entmax
from collections import Counter

# import numpy as np

import torch.nn.functional as F
from feature_map import FeatureMap, train_separation
from synthetic_data import SyntheticDataset
from prettytable import PrettyTable


import argparse


parser = argparse.ArgumentParser(description="Synthetic Metastable State")
parser.add_argument("--D", type=int, default=5, help="dimension of memory pattern")
parser.add_argument("--N", type=int, default=10, help="number of memories")
parser.add_argument("--D_phi", type=int, default=5, help="dimension of feature space")


parser.add_argument(
    "--iteration", type=int, default=20, help="iteration of separation training"
)
parser.add_argument("--lr", type=float, default=1, help="lr of separation training")
parser.add_argument(
    "--tau", type=float, default=0.1, help="separation loss temperature"
)

args = parser.parse_args()


# %%
torch.random.manual_seed(42)

D = args.D
N = args.N
D_phi = args.D_phi
iteration = args.iteration
lr = args.lr

memories = SyntheticDataset(N, D)
queries = SyntheticDataset(N, D)

data_loader = torch.utils.data.DataLoader(
    memories, batch_size=len(memories), shuffle=True
)
data_loader_test = torch.utils.data.DataLoader(
    queries, batch_size=len(queries), shuffle=True
)


def cccp(X, Q, alpha, beta, num_iters):
    Xi = Q  # query
    for _ in range(num_iters):
        P = entmax(X @ Xi * beta, alpha=alpha, dim=0)
        Xi = X.T @ P
    return P


def kernelized_cccp(X, Q, alpha, beta, num_iters, w):
    Xi = Q  # query
    for _ in range(num_iters):
        P = entmax(w(X) @ w(Xi.T).T * beta, alpha=alpha, dim=0)
        Xi = X.T @ P
    return P


num_iters = 20
eps = 1e-2
device = torch.device("cuda:" + "0")
X_train = memories.data.to(device)
X_test = queries.data.to(device)
n_samples = args.N
N = args.N
ctrs_total = []

w = train_separation(data_loader, D, D_phi, iteration, lr, args.tau)
alpha = 1.5

for beta in [1]:
    ctrs = []
    # for alpha in [1, 1.5, 2]:

    P = cccp(X_train, X_test.T, alpha, beta, num_iters)
    eps_ = eps if alpha == 1 else 0
    sizes = (P > eps_).sum(dim=0)

    ctr = Counter(sizes.tolist())
    ctrs.append(ctr)

    P = kernelized_cccp(X_train, X_test.T, alpha, beta, num_iters, w)
    eps_ = eps if alpha == 1 else 0
    sizes = (P > eps_).sum(dim=0)

    ctr = Counter(sizes.tolist())
    ctrs.append(ctr)

    ctrs_total.append(ctrs)

tab = PrettyTable(["K", "beta", "mode", "metastable size"])
for k in range(0, 11):
    for i, beta in enumerate([1]):
        for j, mode in enumerate(["cccp", "k-cccp"]):
            score = round(ctrs_total[i][j][k] / n_samples * 100, 2)
            tab.add_row([k, beta, mode, score])


print(tab)

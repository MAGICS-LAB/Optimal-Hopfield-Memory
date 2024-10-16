import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn.utils.weight_norm as weight_norm

# import geotorch


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


def custom_pdist(X):
    n = X.size(0)
    X_norm = torch.sum(X**2, dim=1, keepdim=True)
    dist_matrix = X_norm - 2 * torch.mm(X, X.t()) + X_norm.t()
    return torch.sqrt(torch.abs(dist_matrix))


def avg_separate(x, tau=0.1):
    x = torch.nn.functional.normalize(x, dim=-1)
    # tgt = x.clone().detach()
    inner = torch.matmul(x, x.T) / tau
    label = torch.tensor([i for i in range(x.size(0))]).to(x.device)
    return F.cross_entropy(inner, label)


def separation_loss(x, tau=0.1):
    # x = torch.nn.functional.normalize(x, dim=-1)
    # t=2
    # return torch.nn.functional.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()
    # x: (batch_size, hid_dim)
    x = torch.nn.functional.normalize(x, dim=-1)
    base = x.size(0) * (x.size(0) - 1) / 2
    tgt = x.clone().detach()
    inner = torch.triu(torch.matmul(x, tgt.T), diagonal=1)
    # print(inner.tolist())
    loss = torch.sum(torch.exp(inner / tau))
    log_loss = torch.log(loss / (base))
    return log_loss / x.size(0)


def train_separation_cont(memory, inp_dim, hid_dim, iteration, lr, tau=1):
    w = FeatureMap(inp_dim, hid_dim).cuda()
    opt = torch.optim.SGD(w.parameters(), lr, momentum=0.9)
    memory = memory.cuda()
    for N in range(iteration):

        opt.zero_grad()

        phi_x = w(memory)
        loss = separation_loss(phi_x, tau)
        loss.backward()
        opt.step()

    print("final loss:", loss.item())
    return w


def train_separation(dataset, inp_dim, hid_dim, iteration, lr, tau=0.1):

    w = FeatureMap(inp_dim, hid_dim).cuda()
    opt = torch.optim.Adam(w.parameters(), lr, weight_decay=0.0)
    loss_track = []

    # w.phi.parametrizations.weight.retain_grad()
    w.phi.weight.retain_grad()

    for N in range(iteration):

        epoch_loss = 0.0
        epoch_step = 0.0

        for x, y in dataset:

            opt.zero_grad()

            x = x.cuda()
            w0 = w.phi.weight
            phi_x = w(x)
            phi_x.retain_grad()
            # loss = phi_x.sum()
            loss = separation_loss(phi_x)
            loss.backward()
            # print(dir(w.phi.parametrizations.weight))
            # # print(dir(w.phi.parametrizations))
            # print(w.phi.parametrizations.weight.modules[0])

            # print(phi_x.grad)
            # print(torch.norm(phi_x[0]))
            opt.step()

            epoch_loss += loss.item()
            epoch_step += 1

        loss_track.append((epoch_loss / epoch_step))
        if N % 20 == 0:
            print("Iteration", N, "loss:", round(loss_track[-1], 3))

    return w


# N = 2
# D = 2

# from synthetic_data import SyntheticDataset

# memory_set = SyntheticDataset(N, D)
# # Create a DataLoader to iterate over the dataset
# data_loader = torch.utils.data.DataLoader(memory_set, batch_size=len(memory_set), shuffle=True)
# train_separation(data_loader, D, D, 100, 0.1, tau=0.1)

import pickle

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, TensorDataset


with open("../algorithms/SM/results/plots/Rotated_75_MNIST_0/Z_out.pkl", "rb") as fp:
    Z_train = pickle.load(fp)

with open("../algorithms/SM/results/plots/Rotated_75_MNIST_0/Z_test.pkl", "rb") as fp:
    Z_test = pickle.load(fp)


class Score_Func(nn.Module):
    def __init__(self, feature_dim=32):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(feature_dim, feature_dim))

    def forward(self, x):
        score = self.fc(x)
        return score


def score_matching(score_net, samples, n_particles=1):
    dup_samples = samples.unsqueeze(0).expand(n_particles, *samples.shape).contiguous().view(-1, *samples.shape[1:])
    dup_samples.requires_grad_(True)

    grad1 = score_net(dup_samples)
    loss1 = torch.norm(grad1, dim=-1) ** 2 / 2.0

    loss2 = torch.zeros(dup_samples.shape[0], device=dup_samples.device)
    for i in range(dup_samples.shape[1]):
        grad = torch.autograd.grad(grad1[:, i].sum(), dup_samples, create_graph=True)[0][:, i]
        loss2 += grad

    loss = loss1 + loss2
    return loss.mean()


score_func = Score_Func().cuda()
density_optimizer = torch.optim.Adam(score_func.parameters(), lr=1e-3)

dataset = TensorDataset(torch.Tensor(Z_train))
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

total_estimation_loss = 0
total_samples = 0

for iteration in range(5000):
    train_iter_loader = iter(train_loader)
    samples = train_iter_loader.next()
    samples = samples[0].cuda()

    density_loss = score_matching(score_func, samples)
    total_estimation_loss += density_loss
    total_samples += len(samples)

    density_optimizer.zero_grad()
    density_loss.backward()
    density_optimizer.step()

    if iteration % 300 == 0:
        print(
            "Train set: Iteration: [{}/{}]\tLoss: {:.6f}".format(
                iteration,
                5000,
                total_estimation_loss / total_samples,
            )
        )

        total_estimation_loss = 0
        total_samples = 0


dataset = TensorDataset(torch.Tensor(Z_test))
test_loader = DataLoader(dataset, batch_size=64, shuffle=True)

Z_adapts = []
with torch.no_grad():
    for iteration, (samples) in enumerate(test_loader):
        samples = samples[0].cuda()

        for i in range(500):
            grad = score_func(samples)
            samples = samples.add(grad * 0.1)

        Z_adapts += samples.tolist()

Z_out = Z_train + Z_adapts
data = np.asarray(Z_out)

tsne_model = TSNE(n_components=2, init="pca")
Z_2d = tsne_model.fit_transform(data)

plt.scatter(Z_2d[: len(Z_train), 0], Z_2d[: len(Z_train), 1], marker=".")
plt.scatter(Z_2d[len(Z_train) :, 0], Z_2d[len(Z_train) :, 1], marker=".")
plt.savefig("out/Z_Score_tSNE.png")

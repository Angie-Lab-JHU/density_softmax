# prerequisites
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.utils import save_image


with open("../algorithms/SM/results/plots/MNIST_0/Z_out.pkl", "rb") as fp:
    Z_out = pickle.load(fp)

with open("../algorithms/SM/results/plots/Rotated_75_MNIST_0/Z_test.pkl", "rb") as fp:
    Z_test = pickle.load(fp)


class VAE(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim):
        super(VAE, self).__init__()

        # encoder part
        self.fc1 = nn.Linear(x_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc31 = nn.Linear(h_dim2, z_dim)
        self.fc32 = nn.Linear(h_dim2, z_dim)
        # decoder part
        self.fc4 = nn.Linear(z_dim, h_dim2)
        self.fc5 = nn.Linear(h_dim2, h_dim1)
        self.fc6 = nn.Linear(h_dim1, x_dim)

    def encoder(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc31(h), self.fc32(h)  # mu, log_var

    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)  # return z sample

    def decoder(self, z):
        h = F.relu(self.fc4(z))
        h = F.relu(self.fc5(h))
        return F.relu(self.fc6(h))

    def forward(self, x):
        mu, log_var = self.encoder(x.view(-1, 32))
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var

    def gaussian_likelihood(self, x_hat, x):
        scale = torch.exp(torch.nn.Parameter(torch.Tensor([0.0])))
        scale = scale.cuda()
        mean = x_hat
        dist = torch.distributions.Normal(mean, scale)

        # measure prob of seeing image under p(x|z)
        log_pxz = dist.log_prob(x)
        return log_pxz


# build model
vae = VAE(x_dim=32, h_dim1=512, h_dim2=256, z_dim=2)
if torch.cuda.is_available():
    vae.cuda()

optimizer = optim.Adam(vae.parameters())
# return reconstruction error + KL divergence losses
def loss_function(recon_x, x, mu, log_var):
    BCE = F.mse_loss(recon_x, x.view(-1, 32), reduction="sum")
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD


trn_x = np.asarray(Z_out)
train_tensor = torch.from_numpy(trn_x).float()
train_dataset = torch.utils.data.TensorDataset(train_tensor)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True)

test_x = np.asarray(Z_test)
test_tensor = torch.from_numpy(test_x).float()
test_dataset = torch.utils.data.TensorDataset(test_tensor)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=True)


def train(epoch):
    vae.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = data[0].cuda()
        optimizer.zero_grad()
        recon_batch, mu, log_var = vae(data)
        loss = loss_function(recon_batch, data, mu, log_var)

        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item() / len(data),
                )
            )
    print("====> Epoch: {} Average loss: {:.4f}".format(epoch, train_loss / len(train_loader.dataset)))


def grad_density(vae, x):
    x.requires_grad_(True)
    recon_batch, mu, log_var = vae(x)
    print(recon_batch)
    quit()
    log_pxz = vae.gaussian_likelihood(recon_batch, x)

    grad = torch.autograd.grad(log_pxz.sum(), x, create_graph=True)[0]
    print(grad.shape)
    quit()

    return grad


def test():
    # vae.eval()
    train_loss = 0
    Z_adapt = []
    for batch_idx, data in enumerate(test_loader):
        z = data[0].cuda()

        for i in range(100):
            grad1 = grad_density(vae, z)
            z = z.add(grad1 * 0.01)

        Z_adapt += z.tolist()


for epoch in range(1, 1):
    train(epoch)

test()

with torch.no_grad():
    z = torch.randn(1000, 2).cuda()
    sample = vae.decoder(z).cuda()
    sample = sample.tolist()
    with open("out/Z_VAE.pkl", "wb") as fp:
        pickle.dump(sample, fp)

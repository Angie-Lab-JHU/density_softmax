import math
import pickle

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics

# prerequisites
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.utils import save_image

class Classifer(nn.Module):
	def __init__(self):
		super().__init__()
		self.fc = nn.Sequential(nn.Linear(2, 2))

	def forward(self, x, prob = None):
		score = self.fc(x)
		if prob == None:
			return score
		return score * prob

class VAE(nn.Module):
	def __init__(self, x_dim, h_dim1, h_dim2, z_dim):
		super(VAE, self).__init__()
		self.x_dim = x_dim

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
		mu, log_var = self.encoder(x.view(-1, self.x_dim))
		z = self.sampling(mu, log_var)
		return self.decoder(z), mu, log_var

	def gaussian_likelihood(self, x_hat, x):
		dist = torch.distributions.Normal(x, torch.tensor([1.0]).cuda())
		# measure prob of seeing image under p(x|z)
		log_pxz = dist.log_prob(x_hat)
		return log_pxz

def create_dataset():
	# Create dataset
	alpha_0 = np.random.uniform(2 / 3 * np.pi, 0.85 * np.pi, 100)
	alpha_1 = np.random.uniform(1.2 * np.pi, 3 / 2 * np.pi, 100)

	rad_tr = np.random.normal(3, 0.5, 100)
	x_tr_0 = (rad_tr * np.sin(alpha_0), rad_tr * np.cos(alpha_0))
	y_tr_0 = np.zeros(100)
	x_tr_1 = (rad_tr * np.sin(alpha_1), rad_tr * np.cos(alpha_1))
	y_tr_1 = np.ones(100)

	x_tr = np.concatenate((x_tr_0, x_tr_1), axis=1)
	x_tr = np.moveaxis(x_tr, 0, -1)
	y_tr = np.concatenate((y_tr_0, y_tr_1))

	rad_te = np.random.normal(10, 0.5, 100)
	noise_te = np.random.normal(0, 0.15, 100)
	x_te = (rad_te * noise_te, rad_te)
	x_te = np.array(x_te)
	x_te = np.moveaxis(x_te, 0, -1)

	y_te = []
	for x in x_te:
		if x[0] > 0:
			y_te.append(0)
		else:
			y_te.append(1)

	y_te = np.array(y_te)

	return x_tr, y_tr, x_te, y_te


def loss_function(recon_x, x, mu, log_var):
	BCE = F.mse_loss(recon_x, x.view(-1, 2), reduction="sum")
	KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
	return BCE + KLD

def train_model(model, optimizer, x_tr, y_tr, epochs, criterion, vae):
	model.cuda()
	model.train()
	samples = torch.tensor(x_tr.astype(np.float32)).cuda()
	labels = torch.tensor(y_tr.astype(np.int64)).cuda()
	for epoch in range(epochs):
		# recon_batch, mu, log_var = vae(samples)
		# prob = torch.exp(vae.gaussian_likelihood(recon_batch, samples))
		# y_hat = model(samples, prob)
		y_hat = model(samples)
		loss = criterion(y_hat, labels)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	model.eval()
	# model.cpu()

def main():
	# Create dataset
	x_tr, y_tr, x_te, y_te = create_dataset()

	vae = VAE(x_dim=2, h_dim1=2, h_dim2=2, z_dim=2)
	if torch.cuda.is_available():
		vae.cuda()
	optimizer = optim.Adam(vae.parameters())

	# Estimate density function
	density_losses = []
	iters = []
	vae.train()

	# score_func.train()
	for ite in range(5000):
		samples = torch.tensor(x_tr.astype(np.float32)).cuda()
		lables = torch.tensor(y_tr.astype(np.int64)).cuda()
		recon_batch, mu, log_var = vae(samples)
		optimizer.zero_grad()
		loss = loss_function(recon_batch, samples, mu, log_var)
		loss.backward()
		optimizer.step()

	# Fast-adaptation on testing dataset
	vae.eval()

	model = Classifer()
	model_optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
	criterion = nn.CrossEntropyLoss()
	train_model(model, model_optimizer, x_tr, y_tr, 5000, criterion, vae)

	# Print accuracies
	samples = torch.tensor(x_te.astype(np.float32)).cuda()
	recon_batch, mu, log_var = vae(samples)
	prob = torch.exp(vae.gaussian_likelihood(recon_batch, samples))
	y_hat = model(samples, prob).cpu()
	_, y_hat = torch.max(y_hat, 1)
	print(sklearn.metrics.accuracy_score(y_hat.detach().numpy(), y_te))

	# Prepare for visualization
	h = 0.02  # point in the mesh [x_min, m_max]x[y_min, y_max].

	x_min, x_max = x_tr[:, 0].min() - 9, x_tr[:, 0].max() + 9
	y_min, y_max = x_tr[:, 1].min() - 6, x_tr[:, 1].max() + 12
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

	# Predictions to obtain the classification results
	samples = torch.tensor(np.c_[xx.ravel(), yy.ravel()].astype(np.float32)).cuda()
	recon_batch, mu, log_var = vae(samples)
	prob = torch.exp(vae.gaussian_likelihood(recon_batch, samples))
	Z = model(samples, prob).cpu()
	_, Z = torch.max(Z, 1)
	Z = Z.reshape(xx.shape)

	# Plotting
	fig, ax = plt.subplots()
	cdict = {0: "violet", 1: "yellow"}
	for g in np.unique(y_te):
		i = np.where(y_te == g)
		ax.scatter(x_te[i, 0], x_te[i, 1], label=g, c=cdict[g])

	plt.gca().add_patch(plt.Circle((0, 0), 3, color="green", fill=False))
	plt.plot([-8, 8], [0, 0], linestyle="--", color="black")
	plt.plot([0, 0], [-8, 8], linestyle="--", color="green")

	plt.contourf(xx, yy, Z, alpha=0.4)
	plt.scatter(x_tr[:, 0], x_tr[:, 1], c=y_tr, alpha=0.8)

	plt.xlabel("Dim-1")
	plt.ylabel("Dim-2")

	plt.savefig("demo.jpg")


if __name__ == "__main__":
	main()

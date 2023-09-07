import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import pkg_resources
import importlib
importlib.reload(pkg_resources)

import matplotlib.pyplot as plt
import matplotlib.colors as colors

import sklearn.datasets

import numpy as np
import tensorflow as tf

import official.nlp.modeling.layers as nlp_layers
from sklearn.neighbors import KernelDensity
import edward2 as ed
import random
# 4
seed_value = 25
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

DEFAULT_X_RANGE = (-3, 3)
DEFAULT_Y_RANGE = (-2, 2)
DEFAULT_CMAP = colors.ListedColormap(["#377eb8", "#ff7f00"])
DEFAULT_NORM = colors.Normalize(vmin=0, vmax=1,)
DEFAULT_N_GRID = 100

def make_training_data(sample_size=500, two_moons = True):
  """Create two moon training dataset."""
  train_examples, train_labels = sklearn.datasets.make_moons(
      n_samples=2 * sample_size, noise=0.1)

  train_examples[train_labels == 0] += [-0.1, 0.2]
  train_examples[train_labels == 1] += [0.1, -0.2]

  if two_moons == False:
    train_examples_1 = np.random.multivariate_normal(
        (-1.5, 0), cov=np.diag((0.01, 0.005)), size=500)
    train_examples_2 = np.random.multivariate_normal(
        (1.5, 0), cov=np.diag((0.01, 0.005)), size=500)
    train_examples = np.concatenate((train_examples_1, train_examples_2), axis=0)
    train_labels = np.concatenate((np.zeros(500), np.ones(500)))

  return train_examples, train_labels

def make_testing_data(x_range=DEFAULT_X_RANGE, y_range=DEFAULT_Y_RANGE, n_grid=DEFAULT_N_GRID):
  """Create a mesh grid in 2D space."""
  # testing data (mesh grid over data space)
  x = np.linspace(x_range[0], x_range[1], n_grid)
  y = np.linspace(y_range[0], y_range[1], n_grid)
  xv, yv = np.meshgrid(x, y)
  return np.stack([xv.flatten(), yv.flatten()], axis=-1)

def make_ood_data(sample_size=500, means=(1, -1.5), vars=(0.1, 0.001)):
  return np.random.multivariate_normal(
      means, cov=np.diag(vars), size=sample_size)

# Load the train, test and OOD datasets.
train_examples, train_labels = make_training_data(
    sample_size=500)
test_examples = make_testing_data()
ood_examples = make_ood_data(sample_size=500)

fig, axs = plt.subplots(1, 2, figsize = (8, 4), constrained_layout=True)

pos_examples = train_examples[train_labels == 0]
neg_examples = train_examples[train_labels == 1]

axs[0].scatter(pos_examples[:, 0], pos_examples[:, 1], c="#377eb8", alpha=0.5)
axs[0].scatter(neg_examples[:, 0], neg_examples[:, 1], c="#ff7f00", alpha=0.5)
axs[0].scatter(ood_examples[:, 0], ood_examples[:, 1], c="red", alpha=0.1)

axs[0].legend(["Positive", "Negative", "Out-of-Domain"])

axs[0].set_ylim(DEFAULT_Y_RANGE)
axs[0].set_xlim(DEFAULT_X_RANGE)
axs[0].set_title("Two moons visualization", fontsize=18)

# Load the train, test and OOD datasets.
train_examples, train_labels = make_training_data(
    sample_size=500, two_moons = False)
test_examples = make_testing_data()
ood_examples = make_ood_data(sample_size=500, means=(0, -1.25), vars=(0.5, 0.001))

pos_examples = train_examples[train_labels == 0]
neg_examples = train_examples[train_labels == 1]

axs[1].scatter(pos_examples[:, 0], pos_examples[:, 1], c="#377eb8", alpha=0.5)
axs[1].scatter(neg_examples[:, 0], neg_examples[:, 1], c="#ff7f00", alpha=0.5)
axs[1].scatter(ood_examples[:, 0], ood_examples[:, 1], c="red", alpha=0.1)

axs[1].legend(["Positive", "Negative", "Out-of-Domain"])

axs[1].set_ylim(DEFAULT_Y_RANGE)
axs[1].set_xlim(DEFAULT_X_RANGE)
axs[1].set_title("Two ovals visualization", fontsize=18)

# plt.tight_layout()
plt.savefig("out/teaser.pdf")
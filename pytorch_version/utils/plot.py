import pickle

import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE


if __name__ == "__main__":

    with open("../algorithms/SM/results/plots/MNIST_0/Z_out.pkl", "rb") as fp:
        Z_train = pickle.load(fp)

    with open("out/Z_VAE.pkl", "rb") as fp:
        Z_sample = pickle.load(fp)

    Z_out = Z_train + Z_sample
    data = np.asarray(Z_out)

    tsne_model = TSNE(n_components=2, init="pca")
    Z_2d = tsne_model.fit_transform(data)

    plt.scatter(Z_2d[: len(Z_train), 0], Z_2d[: len(Z_train), 1], marker=".")
    plt.scatter(Z_2d[len(Z_train) :, 0], Z_2d[len(Z_train) :, 1], marker=".")
    plt.savefig("out/Z_VAE_tSNE.png")

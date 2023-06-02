import pickle

import numpy as np
from sklearn.manifold import TSNE
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity


with open("../algorithms/SM/results/plots/MNIST_0/Z_out.pkl", "rb") as fp:
    Z_out = pickle.load(fp)

with open("../algorithms/SM/results/plots/Rotated_75_MNIST_0/Z_test.pkl", "rb") as fp:
    Z_test = pickle.load(fp)

data = np.asarray(Z_out)[:1000]

# use grid search cross-validation to optimize the bandwidth
params = {"bandwidth": np.logspace(-1, 1, 20)}
grid = GridSearchCV(KernelDensity(), params)
grid.fit(data)

print("best bandwidth: {0}".format(grid.best_estimator_.bandwidth))

# use the best estimator to compute the kernel density estimate
kde = grid.best_estimator_
test_data = kde.sample(1000)
test_data = test_data.tolist()

with open("out/Z_KDE.pkl", "wb") as fp:
    pickle.dump(test_data, fp)

import pickle

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import seaborn as sns
from scipy.stats import wasserstein_distance
from scipy.stats import entropy

def get_entropy(filename):
    entropies = []
    with open(filename, "rb") as fp:
        probs = pickle.load(fp)
    for prob in probs:
        entropies.append(entropy(prob))
    return entropies


entropies_deterministic = get_entropy("out/ood_cifar/deterministic/ood_probs.pkl")
entropies_dropout= get_entropy("out/ood_cifar/dropout/ood_probs.pkl")
entropies_bnn= get_entropy("out/ood_cifar/variational_inference/ood_probs.pkl")
entropies_rank1 = get_entropy("out/ood_cifar/rank1_bnn/ood_probs.pkl")
entropies_sngp = get_entropy("out/ood_cifar/sngp/ood_probs.pkl")
entropies_batchensemble = get_entropy("out/ood_cifar/batchensemble/ood_probs.pkl")
entropies_ensemble = get_entropy("out/ood_cifar/deepensembles/ood_probs.pkl")
entropies_density_softmax = get_entropy("out/ood_cifar/density_softmax/ood_probs.pkl")

# # seaborn histogram
sns.kdeplot(entropies_deterministic, label="ERM")
sns.kdeplot(entropies_dropout, label="MC Dropout")
sns.kdeplot(entropies_bnn, label="MFVI BNN")
sns.kdeplot(entropies_rank1, label="Rank-1 BNN")
sns.kdeplot(entropies_sngp, label="SNGP")
sns.kdeplot(entropies_batchensemble, label="BatchEnsemble")
sns.kdeplot(entropies_ensemble, label="Deep Ensembles")
sns.kdeplot(entropies_density_softmax, label="Density-Softmax", color="blue")
plt.xlabel("Predictive Entropy", fontsize=11)
plt.ylabel("Normalized density", fontsize=11)

plt.legend()
plt.tight_layout()
plt.savefig("ood_predictive_entropy.pdf")
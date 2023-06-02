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

def plot_entropy(entropies, out_entropies, ood_entropies, ax, color, method):
    # ax.set_ylim((0, 14))
    # ax.set_xlim((-0.2, 3))
    # ax.hist(entropies, label="iid",alpha=0.8, density=True, bins = 10)
    # ax.hist(out_entropies, label="ood",alpha=0.8, density=True, bins = 10)

    sns.kdeplot(entropies, color=color, label="CIFAR-10", ax = ax, bw=0.5)
    sns.kdeplot(out_entropies, color=color, label="CIFAR-10.1", linestyle="dotted", ax = ax, bw=0.5)
    sns.kdeplot(ood_entropies, color=color, label="CIFAR-100", linestyle="--", ax = ax, bw=0.5)
    ax.set_title(method, size=11)
    ax.set_xlabel("Predictive Entropy", fontsize=11)
    ax.set_ylabel("Normalized density", fontsize=11)
    ax.legend()

entropies_deterministic = get_entropy("utils/out/flow_cifar10/plots/probs_deterministic.pkl")
entropies_dropout= get_entropy("utils/out/dropout_cifar10/plots/probs.pkl")
entropies_bnn= get_entropy("utils/out/variational_inference_cifar10/plots/probs.pkl")
entropies_rank1 = get_entropy("utils/out/rank1_cifar10/plots/probs.pkl")
entropies_sngp = get_entropy("utils/out/sngp_cifar10/plots/probs.pkl")
entropies_batchensemble = get_entropy("utils/out/batchensemble_cifar10/plots/probs.pkl")
entropies_ensemble = get_entropy("utils/out/ensemble_cifar10/plots/probs.pkl")
entropies_density_softmax = get_entropy("utils/out/flow_cifar10/plots/probs.pkl")

out_entropies_deterministic = get_entropy("utils/out/flow_cifar10/plots/ood_probs_deterministic.pkl")
out_entropies_dropout= get_entropy("utils/out/dropout_cifar10/plots/ood_probs.pkl")
out_entropies_bnn= get_entropy("utils/out/variational_inference_cifar10/plots/ood_probs.pkl")
out_entropies_rank1 = get_entropy("utils/out/rank1_cifar10/plots/ood_probs.pkl")
out_entropies_sngp = get_entropy("utils/out/sngp_cifar10/plots/ood_probs.pkl")
out_entropies_batchensemble = get_entropy("utils/out/batchensemble_cifar10/plots/ood_probs.pkl")
out_entropies_ensemble = get_entropy("utils/out/ensemble_cifar10/plots/ood_probs.pkl")
out_entropies_density_softmax = get_entropy("utils/out/flow_cifar10/plots/ood_probs.pkl")

ood_entropies_deterministic = get_entropy("out/ood_cifar/deterministic/ood_probs.pkl")
ood_entropies_dropout= get_entropy("out/ood_cifar/dropout/ood_probs.pkl")
ood_entropies_bnn= get_entropy("out/ood_cifar/variational_inference/ood_probs.pkl")
ood_entropies_rank1 = get_entropy("out/ood_cifar/rank1_bnn/ood_probs.pkl")
ood_entropies_sngp = get_entropy("out/ood_cifar/sngp/ood_probs.pkl")
ood_entropies_batchensemble = get_entropy("out/ood_cifar/batchensemble/ood_probs.pkl")
ood_entropies_ensemble = get_entropy("out/ood_cifar/deepensembles/ood_probs.pkl")
ood_entropies_density_softmax = get_entropy("out/ood_cifar/density_softmax/ood_probs.pkl")

# # seaborn histogram
fig, axs = plt.subplots(2, 4, figsize = (16, 8), constrained_layout=True)

plot_entropy(entropies_deterministic, out_entropies_deterministic, ood_entropies_deterministic, axs[0][0], "tab:blue", "ERM")
plot_entropy(entropies_dropout, out_entropies_dropout, ood_entropies_dropout, axs[0][1], "tab:orange", "MC Dropout")
plot_entropy(entropies_bnn, out_entropies_bnn, ood_entropies_bnn, axs[0][2], "tab:green", "MFVI BNN")
plot_entropy(entropies_rank1, out_entropies_rank1, ood_entropies_rank1, axs[0][3], "tab:red", "Rank-1 BNN")
plot_entropy(entropies_sngp, out_entropies_sngp, ood_entropies_sngp, axs[1][0], "tab:purple", "SNGP")
plot_entropy(entropies_batchensemble, out_entropies_batchensemble, ood_entropies_batchensemble, axs[1][1], "tab:brown", "BatchEnsemble")
plot_entropy(entropies_ensemble, out_entropies_ensemble, ood_entropies_ensemble, axs[1][2], "tab:pink", "Deep Ensembles")
plot_entropy(entropies_density_softmax, out_entropies_density_softmax, ood_entropies_density_softmax, axs[1][3], "blue", "Density-Softmax")

# plt.xlabel("Predictive Entropy", fontsize=11)
# plt.ylabel("Density", fontsize=11)

# plt.legend()
plt.tight_layout(h_pad=5)
plt.savefig("out/apd_out_predictive_entropy.png")
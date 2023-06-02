import pickle

import matplotlib.pyplot as plt
import numpy as np
# import scipy.stats
# import seaborn as sns
# from scipy.stats import wasserstein_distance


# with open("../algorithms/Flows/results/plots/CIFAR10_1/tr_entropies.pkl", "rb") as fp:
#     test_ERM = pickle.load(fp)
# with open("../algorithms/Flows/results/plots/CIFAR10_1/te_entropies.pkl", "rb") as fp:
#     test_DSM = pickle.load(fp)
# # with open("../algorithms/Flows/results/plots/CIFAR10_1/te_entropies.pkl", "rb") as fp:
# #     test_DSM_wc = pickle.load(fp)

# # # seaborn histogram
# plt.xlabel("Predictive Entropy")
# sns.kdeplot(test_ERM, label="ERM")
# sns.kdeplot(test_DSM, label="DSM")
# # sns.kdeplot(test_DSM_wc, label="DSMwc")
# plt.title("Out of distribution - MNIST", size=20)

# plt.legend()
# plt.savefig("predictive_entropy.png")


with open("utils/out/flow_cifar10/plots/train_nll.pkl", "rb") as fp:
    train_NLL = pickle.load(fp)

with open("utils/out/flow_cifar10/plots/test_nll.pkl", "rb") as fp:
    test_in = pickle.load(fp)

with open("utils/out/flow_cifar10/plots/out_1_nll.pkl", "rb") as fp:
    out_1_nll = pickle.load(fp)

with open("utils/out/flow_cifar10/plots/out_2_nll.pkl", "rb") as fp:
    out_2_nll = pickle.load(fp)

with open("utils/out/flow_cifar10/plots/out_3_nll.pkl", "rb") as fp:
    out_3_nll = pickle.load(fp)

with open("utils/out/flow_cifar10/plots/out_4_nll.pkl", "rb") as fp:
    out_4_nll = pickle.load(fp)

with open("utils/out/flow_cifar10/plots/out_5_nll.pkl", "rb") as fp:
    out_5_nll = pickle.load(fp)

# with open("../algorithms/VAE/results/plots/MNIST_1/te_out_nlls.pkl", "rb") as fp:
#     test_zigzac = pickle.load(fp)

bins = int(100 / 1)
plt.xlabel("Likelihood value p(z)", fontsize=11)
plt.ylabel("Normalized density", fontsize=11)
plt.xlim([0.4, 1])
plt.hist(train_NLL, label="train", alpha=0.8,density=True, bins=bins)
plt.hist(test_in, label="test_iid", alpha=0.8,density=True, bins=bins)
plt.hist(out_1_nll, label="ood_1", alpha=0.8,density=True, bins=bins)
plt.hist(out_2_nll, label="ood_2",alpha=0.8, density=True, bins=bins)
plt.hist(out_3_nll, label="ood_3", alpha=0.8,density=True, bins=bins)
plt.hist(out_4_nll, label="ood_4", alpha=0.8,density=True, bins=bins)
plt.hist(out_5_nll, label="ood_5",alpha=0.8, density=True, bins=bins)
plt.legend()
plt.tight_layout()
plt.savefig("cifar-10-density_histogram.pdf")

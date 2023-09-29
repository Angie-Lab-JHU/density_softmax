import pickle

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt


def calculate_ece(softmaxes, labels, n_bins=10):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    confidences, predictions = torch.max(softmaxes, 1)
    accuracies = predictions.eq(labels)

    ece = torch.zeros(1, device=softmaxes.device)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Calculated |confidence - accuracy| in each bin
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece.item()


def make_model_diagrams(method, ax, softmaxes, labels, n_bins=15):
    """
    outputs - a torch tensor (size n x num_classes) with the outputs from the final linear layer
    - NOT the softmaxes
    labels - a torch tensor (size n) with the labels
    """
    # softmaxes = torch.nn.functional.softmax(outputs, 1)
    softmaxes = torch.tensor(softmaxes)
    labels = torch.tensor(labels)
    confidences, predictions = torch.max(softmaxes, 1)
    accuracies = torch.eq(predictions, labels)
    overall_accuracy = (predictions == labels).sum().item() / len(labels)

    # Reliability diagram
    bins = torch.linspace(0, 1, n_bins + 1)
    width = 1.0 / n_bins
    bin_centers = np.linspace(0, 1.0 - width, n_bins) + width / 2
    bin_indices = [
        confidences.ge(bin_lower) * confidences.lt(bin_upper) for bin_lower, bin_upper in zip(bins[:-1], bins[1:])
    ]

    bin_corrects = np.array([torch.mean(accuracies[bin_index].float()) for bin_index in bin_indices])
    bin_scores = np.array([torch.mean(confidences[bin_index].float()) for bin_index in bin_indices])

    # ax.figure(0, figsize=(8, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    gap = bin_scores - bin_corrects
    confs = ax.bar(bin_centers, bin_corrects, width=width, alpha=0.1, ec="black")
    gaps = ax.bar(
        bin_centers,
        (bin_scores - bin_corrects),
        bottom=bin_corrects,
        color=[1, 0.7, 0.7],
        alpha=0.5,
        width=width,
        hatch="//",
        edgecolor="r",
    )
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.legend([confs, gaps], ["Outputs", "Gap"], loc="best")

    ece = calculate_ece(softmaxes, labels)

    ax.set_ylabel("Accuracy (P[y])", size=11)
    ax.set_xlabel("Confidence", size=11)
    ax.set_title(method, size=11)

    bbox_props = dict(boxstyle="round", fc="lightgrey", ec="brown", lw=2)
    ax.text(0.15, 0.95, "ECE: {:.2f}".format(ece), ha="center", va="center", size=11, weight="bold", bbox=bbox_props)


with open("utils/out/flow_cifar10/plots/ood_probs_deterministic.pkl", "rb") as fp:
    erm_outputs = pickle.load(fp)
with open("utils/out/flow_cifar10/plots/ood_labels_deterministic.pkl", "rb") as fp:
    erm_labels = pickle.load(fp)
with open("utils/out/rank1_cifar10/plots/ood_probs.pkl", "rb") as fp:
    rank1_outputs = pickle.load(fp)
with open("utils/out/rank1_cifar10/plots/ood_labels.pkl", "rb") as fp:
    rank1_labels = pickle.load(fp)
with open("utils/out/ensemble_cifar10/plots/ood_probs.pkl", "rb") as fp:
    ensemble_outputs = pickle.load(fp)
with open("utils/out/ensemble_cifar10/plots/ood_labels.pkl", "rb") as fp:
    ensemble_labels = pickle.load(fp)
with open("utils/out/flow_cifar10/plots/ood_probs.pkl", "rb") as fp:
    density_softmax_outputs = pickle.load(fp)
with open("utils/out/flow_cifar10/plots/ood_labels.pkl", "rb") as fp:
    density_softmax_labels = pickle.load(fp)

# labels = labels.int()
# outputs = outputs.float()
fig, axs = plt.subplots(1, 4, figsize = (16, 4), constrained_layout=True)
make_model_diagrams("ERM", axs[0], erm_outputs, erm_labels)

make_model_diagrams("Rank-1 BNN", axs[1], rank1_outputs, rank1_labels)

make_model_diagrams("Deep Ensembles", axs[2], ensemble_outputs, ensemble_labels)
make_model_diagrams("Density-Softmax", axs[3], density_softmax_outputs, density_softmax_labels)

# plt.title(method + "-" + test_type.lower(), size=30)
plt.tight_layout()
plt.savefig(
    "out/ece.pdf"
)
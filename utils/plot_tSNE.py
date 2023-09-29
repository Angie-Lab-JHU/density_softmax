
import pickle

from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import torch
import numpy as np

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

def plot_TNSE(method, ax, X_2d_tr, tr_labels, label_target_names, acc):
    colors = ["red", "green", "blue", "black", "brown", "grey", "orange", "yellow", "pink", "cyan", "magenta"]
    for i, label in zip(range(len(label_target_names)), label_target_names):
        ax.scatter(X_2d_tr[tr_labels == i, 0], X_2d_tr[tr_labels == i, 1], c=colors[i], marker=".", label=label)
    
    ax.set_ylabel("Principal Component 2", size=11)
    ax.set_xlabel("Principal Component 1", size=11)
    ax.set_title(method, size=11)

    bbox_props = dict(boxstyle="round", fc="lightgrey", ec="brown", lw=2)
    ax.text(0.15, 0.95, "Acc: {:.1f}".format(acc), size=11, weight="bold", bbox=bbox_props, ha='center', va='center', transform=ax.transAxes)

def unique(list1):
    unique_list = []
    for x in list1:
        if x not in unique_list:
            unique_list.append(x)
    return unique_list

tsne_model = TSNE(n_components=2, init="pca")

with open("tSNE/ds/defocus_blur_5_probs.pkl", "rb") as fp:
	latents = pickle.load(fp)

with open("tSNE/ds/defocus_blur_5_labels.pkl", "rb") as fp:
	labels = pickle.load(fp)

Z_2d_erm = tsne_model.fit_transform(latents)

with open("tSNE/ds/shot_noise_1_probs.pkl", "rb") as fp:
	latents = pickle.load(fp)

with open("tSNE/ds/shot_noise_1_labels.pkl", "rb") as fp:
	labels = pickle.load(fp)

Z_2d_ds = tsne_model.fit_transform(latents)

label_target_names = unique(labels)

with open("out/density_softmax/cifar10/model4/clean_probs.pkl", "rb") as fp:
	w_outputs = pickle.load(fp)

with open("out/density_softmax/cifar10/model4/clean_labels.pkl", "rb") as fp:
	w_labels = pickle.load(fp)

with open("out/deterministic/cifar10/model1/clean_probs.pkl", "rb") as fp:
	wo_outputs = pickle.load(fp)

with open("out/deterministic/cifar10/model1/clean_labels.pkl", "rb") as fp:
	wo_labels = pickle.load(fp)

fig, axs = plt.subplots(1, 4, figsize = (16, 4), constrained_layout=True)
plot_TNSE("Without 1-lipschitz", axs[0], Z_2d_erm, labels, label_target_names, 78.6)
plot_TNSE("With 1-lipschitz", axs[1], Z_2d_ds, labels, label_target_names, 91.2)
make_model_diagrams("Without Density", axs[2], wo_outputs, wo_labels)
make_model_diagrams("With Density", axs[3], w_outputs, w_labels)

plt.tight_layout()
plt.savefig(
    "out.pdf"
)
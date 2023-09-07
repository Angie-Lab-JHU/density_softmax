import argparse
import os
import pickle

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE


def plot_TNSE(X_2d_tr, tr_labels, label_target_names, filename):
    colors = ["red", "green", "blue", "black", "brown", "grey", "orange", "yellow", "pink", "cyan", "magenta"]
    plt.figure(figsize=(16, 16))
    for i, label in zip(range(len(label_target_names)), label_target_names):
        plt.scatter(X_2d_tr[tr_labels == i, 0], X_2d_tr[tr_labels == i, 1], c=colors[i], marker=".", label=label)

    plt.savefig(filename)


def unique(list1):
    unique_list = []
    for x in list1:
        if x not in unique_list:
            unique_list.append(x)
    return unique_list


def tsne_plot(Z_train, labels, domain_labels, dir_name, out):
    labels = np.asarray(labels)
    domain_labels = np.asarray(domain_labels)
    label_target_names = unique(labels)
    domain_label_target_names = unique(domain_labels)

    tsne_model = TSNE(n_components=2, init="pca")
    Z_train = np.array(Z_train)
    Z_2d = tsne_model.fit_transform(Z_train)

    plot_TNSE(Z_2d, labels, label_target_names, dir_name + "Z_class_tSNE" + str(out) + ".png")
    plot_TNSE(Z_2d, domain_labels, domain_label_target_names, dir_name + "Z_domain_tSNE" + str(out) + ".png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--plotdir", help="Path to configuration file")
    bash_args = parser.parse_args()
    dir_name = bash_args.plotdir

    with open(dir_name + "Z_train.pkl", "rb") as fp:
        Z_train = pickle.load(fp)
    with open(dir_name + "Y_train.pkl", "rb") as fp:
        Y_train = pickle.load(fp)
    Y_domain_out = [0] * len(Z_train)

    with open(dir_name + "Z_test.pkl", "rb") as fp:
        Z_test = pickle.load(fp)

    with open(dir_name + "Z_adapt.pkl", "rb") as fp:
        Z_adapt = pickle.load(fp)

    with open(dir_name + "Y_test.pkl", "rb") as fp:
        Y_test = pickle.load(fp)

    Y_domain_test = [1] * len(Z_test)

    # Change label of target domain from -1 to #source_domains + 1
    Y_domain_label = len(unique(Y_domain_out))
    for i in range(len(Y_domain_test)):
        Y_domain_test[i] = Y_domain_label

    Z_out_1 = Z_train + Z_test
    Z_out_2 = Z_train + Z_adapt
    Y_train += Y_test
    Y_domain_out += Y_domain_test

    tsne_plot(Z_out_1, Y_train, Y_domain_out, dir_name, out=1)

    tsne_plot(Z_out_2, Y_train, Y_domain_out, dir_name, out=2)

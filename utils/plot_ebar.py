import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import robustness_metrics as rm
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


x = [0, 1, 2, 3, 4, 5]

corruption_types = [
		'gaussian_noise',
		'shot_noise',
		'impulse_noise',
		'defocus_blur',
		'frosted_glass_blur',
		'motion_blur',
		'zoom_blur',
		'snow',
		'frost',
		'fog',
		'brightness',
		'contrast',
		'elastic',
		'pixelate',
		'jpeg_compression',
	]

metrics = {'test/negative_log_likelihood': tf.keras.metrics.Mean(),
			'test/accuracy': tf.keras.metrics.SparseCategoricalAccuracy(),
			'test/ece': rm.metrics.ExpectedCalibrationError(num_bins=15),}

def get_results(method, seed, ood_type):
	with open("out/" + method + "/cifar10/model" + str(seed) + "/" + ood_type + "_labels.pkl", "rb") as fp:
		labels = pickle.load(fp)
	with open("out/" + method + "/cifar10/model" + str(seed) + "/" + ood_type + "_probs.pkl", "rb") as fp:
		probs = pickle.load(fp)
	negative_log_likelihood = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, probs))
	metrics['test/negative_log_likelihood'].update_state(negative_log_likelihood)
	metrics['test/accuracy'].update_state(labels, probs)
	metrics['test/ece'].add_batch(probs, label=labels)
	return metrics['test/negative_log_likelihood'].result().numpy(), metrics['test/accuracy'].result().numpy() * 100, metrics['test/ece'].result()['ece']

def create_skewgraph(method):
	nll, nll_err, acc, acc_err, ece, ece_err = [], [], [], [], [], []

	seed_nll, seed_acc, seed_ece = [], [], []
	for seed in range(1, 10):
		nll_clean, acc_clean, ece_clean = get_results(method, seed, "clean")
		for metric in metrics.values():
			metric.reset_states()
		seed_nll.append(np.mean(nll_clean))
		seed_acc.append(np.mean(acc_clean))
		seed_ece.append(np.mean(ece_clean))

	if method == "posterior_network":
		nll.append(np.mean(seed_nll))
		nll_err.append(np.std(seed_nll)/8)
		acc.append(np.mean(seed_acc))
		acc_err.append(np.std(seed_acc)/8)
		ece.append(np.mean(seed_ece))
		ece_err.append(np.std(seed_ece)/8)
	else:
		nll.append(np.mean(seed_nll))
		nll_err.append(np.std(seed_nll))
		acc.append(np.mean(seed_acc))
		acc_err.append(np.std(seed_acc))
		ece.append(np.mean(seed_ece))
		ece_err.append(np.std(seed_ece))

	for level in range(1, 6):
		seed_nll, seed_acc, seed_ece = [], [], []
		for seed in range(1, 10):
			corrupt_nll, corrupt_acc, corrupt_ece = [], [], []
			for corruption_type in corruption_types:
				nll_tmp, acc_tmp, ece_tmp = get_results(method, seed, f'{corruption_type}_{level}')
				corrupt_nll.append(nll_tmp)
				corrupt_acc.append(acc_tmp)
				corrupt_ece.append(ece_tmp)
				for metric in metrics.values():
					metric.reset_states()
			seed_nll.append(np.mean(corrupt_nll))
			seed_acc.append(np.mean(corrupt_acc))
			seed_ece.append(np.mean(corrupt_ece))
			
		if method == "posterior_network":
			nll.append(np.mean(seed_nll))
			nll_err.append(np.std(seed_nll)/8)
			acc.append(np.mean(seed_acc))
			acc_err.append(np.std(seed_acc)/8)
			ece.append(np.mean(seed_ece))
			ece_err.append(np.std(seed_ece)/8)
		else:
			nll.append(np.mean(seed_nll))
			nll_err.append(np.std(seed_nll))
			acc.append(np.mean(seed_acc))
			acc_err.append(np.std(seed_acc))
			ece.append(np.mean(seed_ece))
			ece_err.append(np.std(seed_ece))

	return nll, nll_err, acc, acc_err, ece, ece_err

methods_dict = {
		"deterministic": "ERM",
		"dropout": "MC Dropout",
		"variational_inference": "MFVI BNN",
		"rank1_bnn": "Rank-1 BNN",
		"sngp": "SNGP",
		"batchensemble": "BatchEnsemble",
		"deepensembles": "Deep Ensembles",
		"mimo": "MIMO",
		"density_softmax": "Density-Softmax",
		"posterior_network": "Posterior Network",
		"heteroscedastic": "Heteroscedastic"
	}

y_erm, yerr_erm, a_erm, aerr_erm, e_erm, eerr_erm = create_skewgraph("deterministic")
y_dropout, yerr_dropout, a_dropout, aerr_dropout, e_dropout, eerr_dropout = create_skewgraph("dropout")
y_vi, yerr_vi, a_vi, aerr_vi, e_vi, eerr_vi = create_skewgraph("variational_inference")
y_rank1, yerr_rank1, a_rank1, aerr_rank1, e_rank1, eerr_rank1 = create_skewgraph("rank1_bnn")
y_sngp, yerr_sngp, a_sngp, aerr_sngp, e_sngp, eerr_sngp = create_skewgraph("batchensemble")
y_bensemble, yerr_bensemble, a_bensemble, aerr_bensemble, e_bensemble, eerr_bensemble = create_skewgraph("batchensemble")
y_ensembles, yerr_ensembles, a_ensembles, aerr_ensembles, e_ensembles, eerr_ensembles = create_skewgraph("deepensembles")
y_mimo, yerr_mimo, a_mimo, aerr_mimo, e_mimo, eerr_mimo = create_skewgraph("mimo")
y_poste, yerr_poste, a_poste, aerr_poste, e_poste, eerr_poste = create_skewgraph("posterior_network")
y_heter, yerr_heter, a_heter, aerr_heter, e_heter, eerr_heter = create_skewgraph("heteroscedastic")
y_dsm, yerr_dsm, a_dsm, aerr_dsm, e_dsm, eerr_dsm = create_skewgraph("density_softmax")


fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

ax1.errorbar(x, y_erm, yerr=yerr_erm, capsize=5, label="ERM", ecolor = "tab:blue", color = "tab:blue")
ax1.errorbar(x, y_dropout, yerr=yerr_dropout, capsize=5, label="MC Dropout", ecolor = "tab:orange", color = "tab:orange")
ax1.errorbar(x, y_vi, yerr=yerr_vi, capsize=5, label="MFVI BNN", ecolor = "tab:green", color = "tab:green")
ax1.errorbar(x, y_rank1, yerr=yerr_rank1, capsize=5, label="Rank-1 BNN", ecolor = "tab:red", color = "tab:red")
ax1.errorbar(x, y_poste, yerr=yerr_poste, capsize=5, label="Posterior Network", ecolor = "#bcbd22", color = "#bcbd22")
ax1.errorbar(x, y_heter, yerr=yerr_heter, capsize=5, label="Heteroscedastic", ecolor = "#17becf", color = "#17becf")
ax1.errorbar(x, y_sngp, yerr=yerr_sngp, capsize=5, label="SNGP", ecolor = "tab:purple", color = "tab:purple")
ax1.errorbar(x, y_mimo, yerr=yerr_mimo, capsize=5, label="MIMO", ecolor = "#7f7f7f", color = "#7f7f7f")
ax1.errorbar(x, y_bensemble, yerr=yerr_bensemble, capsize=5, label="BatchEnsemble", ecolor = "tab:brown", color = "tab:brown")
ax1.errorbar(x, y_ensembles, yerr=yerr_ensembles, capsize=5, label="Deep Ensembles", ecolor = "tab:pink", color = "tab:pink")
ax1.errorbar(x, y_dsm, yerr=yerr_dsm, capsize=5, label="Density-Softmax", ecolor = "blue", color = "blue")
ax1.set(xlabel="Shift Intensity", ylabel="Negative Log-Likelihood")
ax1.set_xticks(x)

ax2.errorbar(x, a_erm, yerr=aerr_erm, capsize=5, label="ERM", ecolor = "tab:blue", color = "tab:blue")
ax2.errorbar(x, a_dropout, yerr=aerr_dropout, capsize=5, label="MC Dropout", ecolor = "tab:orange", color = "tab:orange")
ax2.errorbar(x, a_vi, yerr=aerr_vi, capsize=5, label="MFVI BNN", ecolor = "tab:green", color = "tab:green")
ax2.errorbar(x, a_rank1, yerr=aerr_rank1, capsize=5, label="Rank-1 BNN", ecolor = "tab:red", color = "tab:red")
ax2.errorbar(x, a_poste, yerr=aerr_poste, capsize=5, label="Posterior Network", ecolor = "#bcbd22", color = "#bcbd22")
ax2.errorbar(x, a_heter, yerr=aerr_heter, capsize=5, label="Heteroscedastic", ecolor = "#17becf", color = "#17becf")
ax2.errorbar(x, a_sngp, yerr=aerr_sngp, capsize=5, label="SNGP", ecolor = "tab:purple", color = "tab:purple")
ax2.errorbar(x, a_mimo, yerr=aerr_mimo, capsize=5, label="MIMO", ecolor = "#7f7f7f", color = "#7f7f7f")
ax2.errorbar(x, a_bensemble, yerr=aerr_bensemble, capsize=5, label="BatchEnsemble", ecolor = "tab:brown", color = "tab:brown")
ax2.errorbar(x, a_ensembles, yerr=aerr_ensembles, capsize=5, label="Deep Ensembles", ecolor = "tab:pink", color = "tab:pink")
ax2.errorbar(x, a_dsm, yerr=aerr_dsm, capsize=5, label="Density-Softmax", ecolor = "blue", color = "blue")
ax2.set(xlabel="Shift Intensity", ylabel="Accuracy")
ax2.set_xticks(x)


ax3.errorbar(x, e_erm, yerr=eerr_erm, capsize=5, label="ERM", ecolor = "tab:blue", color = "tab:blue")
ax3.errorbar(x, e_dropout, yerr=eerr_dropout, capsize=5, label="MC Dropout", ecolor = "tab:orange", color = "tab:orange")
ax3.errorbar(x, e_vi, yerr=eerr_vi, capsize=5, label="MFVI BNN", ecolor = "tab:green", color = "tab:green")
ax3.errorbar(x, e_rank1, yerr=eerr_rank1, capsize=5, label="Rank-1 BNN", ecolor = "tab:red", color = "tab:red")
ax3.errorbar(x, e_poste, yerr=eerr_poste, capsize=5, label="Posterior Network", ecolor = "#bcbd22", color = "#bcbd22")
ax3.errorbar(x, e_heter, yerr=eerr_heter, capsize=5, label="Heteroscedastic", ecolor = "#17becf", color = "#17becf")
ax3.errorbar(x, e_sngp, yerr=eerr_sngp, capsize=5, label="SNGP", ecolor = "tab:purple", color = "tab:purple")
ax3.errorbar(x, e_mimo, yerr=eerr_mimo, capsize=5, label="MIMO", ecolor = "#7f7f7f", color = "#7f7f7f")
ax3.errorbar(x, e_bensemble, yerr=eerr_bensemble, capsize=5, label="BatchEnsemble", ecolor = "tab:brown", color = "tab:brown")
ax3.errorbar(x, e_ensembles, yerr=eerr_ensembles, capsize=5, label="Deep Ensembles", ecolor = "tab:pink", color = "tab:pink")
ax3.errorbar(x, e_dsm, yerr=eerr_dsm, capsize=5, label="Density-Softmax", ecolor = "blue", color = "blue")
ax3.set(xlabel="Shift Intensity", ylabel="Expected Calibration Error")
ax3.set_xticks(x)


# Put a legend below current axis
ax2.legend(loc="upper center", bbox_to_anchor=(0.35, -0.25), fancybox=True, shadow=True, ncol=4)

plt.subplots_adjust(left=0.1, bottom=0.3, right=0.98, top=0.7, wspace=0.4, hspace=0.4)
plt.savefig("ebar.pdf", bbox_inches="tight")
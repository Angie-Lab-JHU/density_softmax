import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import pickle
import robustness_metrics as rm
import tensorflow as tf

corruption_types = [
		'brightness',
		'contrast',
		'defocus_blur',
		'elastic_transform',
		'fog',
		'frost',
		'glass_blur',  # Called frosted_glass_blur in CIFAR-10.
		'gaussian_blur',
		'gaussian_noise',
		'impulse_noise',
		'jpeg_compression',
		'pixelate',
		'saturate',
		'shot_noise',
		'spatter',
		'speckle_noise',  # Does not exist for CIFAR-10.
		'zoom_blur',
		]

metrics = {'test/negative_log_likelihood': tf.keras.metrics.Mean(),
			'test/accuracy': tf.keras.metrics.SparseCategoricalAccuracy(),
			'test/ece': rm.metrics.ExpectedCalibrationError(num_bins=15),}

def get_results(method, ood_type):
	with open("out/tmp_cifar100/"+ method + "/" + ood_type + "_labels.pkl", "rb") as fp:
		labels = pickle.load(fp)
	with open("out/tmp_cifar100/"+ method + "/" + ood_type + "_probs.pkl", "rb") as fp:
		probs = pickle.load(fp)
	negative_log_likelihood = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, probs))
	metrics['test/negative_log_likelihood'].update_state(negative_log_likelihood)
	metrics['test/accuracy'].update_state(labels, probs)
	metrics['test/ece'].add_batch(probs, label=labels)
	return metrics['test/negative_log_likelihood'].result().numpy(), metrics['test/accuracy'].result().numpy() * 100, metrics['test/ece'].result()['ece']
	

def create_skewgraph(level):
	key, nll, acc, ece, skew = [], [], [], [], []
	methods_dict = {
		"deterministic": "ERM",
		"dropout": "MC Dropout",
		"variational_inference": "MFVI BNN",
		"rank1_bnn": "Rank-1 BNN",
		"posterior_network": "Posterior Network",
		"heteroscedastic": "Heteroscedastic",
		"sngp": "SNGP",
		"mimo": "MIMO",
		"batchensemble": "BatchEnsemble",
		"deepensembles": "Deep Ensembles",
		"density_softmax": "Density-Softmax"

	}
	for method in methods_dict:
		for corruption_type in corruption_types:
			nll_tmp, acc_tmp, ece_tmp = get_results(method, f'{corruption_type}_{level}')
			nll.append(nll_tmp)
			acc.append(acc_tmp)
			ece.append(ece_tmp)
			for metric in metrics.values():
				metric.reset_states()
		
		key += [methods_dict[method]] * len(corruption_types)
		skew += [level] * len(corruption_types)
	return key, nll, acc, ece, skew 


key0, skew0 = ["ERM", "MC Dropout", "MFVI BNN", "Rank-1 BNN", "Posterior Network", "Heteroscedastic", "SNGP", "MIMO", "BatchEnsemble", "Deep Ensembles", "Density-Softmax"], ["Test", "Test", "Test", "Test", "Test", "Test", "Test", "Test", "Test", "Test", "Test"]

nll_erm, acc_erm, ece_erm = get_results("deterministic", "clean")
for metric in metrics.values():
	metric.reset_states()
nll_dropout, acc_dropout, ece_dropout = get_results("dropout", "clean")
for metric in metrics.values():
	metric.reset_states()
nll_bnn, acc_bnn, ece_bnn = get_results("variational_inference", "clean")
for metric in metrics.values():
	metric.reset_states()
nll_rank1, acc_rank1, ece_rank1 = get_results("rank1_bnn", "clean")
for metric in metrics.values():
	metric.reset_states()
nll_sngp, acc_sngp, ece_sngp = get_results("sngp", "clean")
for metric in metrics.values():
	metric.reset_states()
nll_batchensemble, acc_batchensemble, ece_batchensemble = get_results("batchensemble", "clean")
for metric in metrics.values():
	metric.reset_states()
nll_deepensembles, acc_deepensembles, ece_deepensembles= get_results("deepensembles", "clean")
for metric in metrics.values():
	metric.reset_states()
nll_density_softmax, acc_density_softmax, ece_density_softmax= get_results("density_softmax", "clean")
for metric in metrics.values():
	metric.reset_states()
nll_mimo, acc_mimo, ece_mimo= get_results("mimo", "clean")
for metric in metrics.values():
	metric.reset_states()
nll_posterior_network, acc_posterior_network, ece_posterior_network= get_results("posterior_network", "clean")
for metric in metrics.values():
	metric.reset_states()
nll_heteroscedastic, acc_heteroscedastic, ece_heteroscedastic= get_results("heteroscedastic", "clean")
for metric in metrics.values():
	metric.reset_states()

nll0 = [nll_erm, nll_dropout, nll_bnn, nll_rank1, nll_posterior_network, nll_heteroscedastic, nll_sngp, nll_mimo, nll_batchensemble, nll_deepensembles, nll_density_softmax]
acc0 = [acc_erm, acc_dropout, acc_bnn, acc_rank1, acc_posterior_network, acc_heteroscedastic, acc_sngp, acc_mimo, acc_batchensemble, acc_deepensembles, acc_density_softmax]
ece0 = [ece_erm, ece_dropout, ece_bnn, ece_rank1, ece_posterior_network, ece_heteroscedastic, ece_sngp, ece_mimo, ece_batchensemble, ece_deepensembles, ece_density_softmax]

key1, nll1, acc1, ece1, skew1 = create_skewgraph(level=1)
key2, nll2, acc2, ece2, skew2 = create_skewgraph(level=2)
key3, nll3, acc3, ece3, skew3 = create_skewgraph(level=3)
key4, nll4, acc4, ece4, skew4 = create_skewgraph(level=4)
key5, nll5, acc5, ece5, skew5 = create_skewgraph(level=5)
d = {
	"Method": key0 + key1 + key2 + key3 + key4 + key5,
	"nll": nll0 + nll1 + nll2 + nll3 + nll4 + nll5,
	"acc": acc0 + acc1 + acc2 + acc3 + acc4 + acc5,
	"ece": ece0 + ece1 + ece2 + ece3 + ece4 + ece5,
	"skew": skew0 + skew1 + skew2 + skew3 + skew4 + skew5,
}
df = pd.DataFrame(data=d)

fig, axs = plt.subplots(4, 1, figsize=(20, 20), constrained_layout=True)
plt.rcParams.update({"font.size": 11})

my_pal = {"ERM": "tab:blue", "MC Dropout":"tab:orange", "MFVI BNN":"tab:green", "Rank-1 BNN":"tab:red", 
	"SNGP":"tab:purple", "BatchEnsemble":"tab:brown", "Deep Ensembles":"tab:pink",
	"MIMO": "#7f7f7f", "Posterior Network": "#bcbd22", "Heteroscedastic": "#17becf", "Density-Softmax":"blue"}

earth = plt.imread('out/dataset-skews.png')
axs[0].imshow(earth)
axs[0].axes.xaxis.set_ticklabels([])
axs[0].axes.yaxis.set_ticklabels([])
axs[0].set_xlabel('Shift Intensity', fontsize = 20)

sns0 = sns.boxplot(data=df, x="skew", y="nll", hue="Method", ax=axs[1], palette=my_pal)
sns0.set_xlabel('Shift Intensity', fontsize = 20)
sns0.set_ylabel('Negative Log-Likelihood', fontsize = 20)

sns1 = sns.boxplot(data=df, x="skew", y="acc", hue="Method", ax=axs[2], palette=my_pal)
sns1.set_xlabel('Shift Intensity', fontsize = 20)
sns1.set_ylabel('Accuracy', fontsize = 20)

sns2 = sns.boxplot(data=df, x="skew", y="ece", hue="Method", ax=axs[3], palette=my_pal)
sns2.set_xlabel('Shift Intensity', fontsize = 20)
sns2.set_ylabel('Expected Calibration Error', fontsize = 20)

for line in axs[1].get_lines()[:5]:
    line.set_color("tab:blue")
    line.set_linewidth(2)
for line in axs[1].get_lines()[10:15]:
    line.set_color("tab:orange")
    line.set_linewidth(2)
for line in axs[1].get_lines()[15:20]:
    line.set_color("tab:green")
    line.set_linewidth(2)
for line in axs[1].get_lines()[20:25]:
    line.set_color("tab:red")
    line.set_linewidth(2)
for line in axs[1].get_lines()[25:30]:
    line.set_color("#bcbd22")
    line.set_linewidth(2)
for line in axs[1].get_lines()[30:35]:
    line.set_color("#17becf")
    line.set_linewidth(2)
for line in axs[1].get_lines()[40:45]:
    line.set_color("tab:purple")
    line.set_linewidth(2)
for line in axs[1].get_lines()[45:50]:
    line.set_color("#7f7f7f")
    line.set_linewidth(2)
for line in axs[1].get_lines()[50:55]:
    line.set_color("tab:brown")
    line.set_linewidth(2)
for line in axs[1].get_lines()[55:60]:
    line.set_color("tab:pink")
    line.set_linewidth(2)
for line in axs[1].get_lines()[60:65]:
    line.set_color("blue")
    line.set_linewidth(2)

for line in axs[2].get_lines()[:5]:
    line.set_color("tab:blue")
    line.set_linewidth(2)
for line in axs[2].get_lines()[10:15]:
    line.set_color("tab:orange")
    line.set_linewidth(2)
for line in axs[2].get_lines()[15:20]:
    line.set_color("tab:green")
    line.set_linewidth(2)
for line in axs[2].get_lines()[20:25]:
    line.set_color("tab:red")
    line.set_linewidth(2)
for line in axs[2].get_lines()[25:30]:
    line.set_color("#bcbd22")
    line.set_linewidth(2)
for line in axs[2].get_lines()[30:35]:
    line.set_color("#17becf")
    line.set_linewidth(2)
for line in axs[2].get_lines()[40:45]:
    line.set_color("tab:purple")
    line.set_linewidth(2)
for line in axs[2].get_lines()[45:50]:
    line.set_color("#7f7f7f")
    line.set_linewidth(2)
for line in axs[2].get_lines()[50:55]:
    line.set_color("tab:brown")
    line.set_linewidth(2)
for line in axs[2].get_lines()[55:60]:
    line.set_color("tab:pink")
    line.set_linewidth(2)
for line in axs[2].get_lines()[60:65]:
    line.set_color("blue")
    line.set_linewidth(2)

for line in axs[3].get_lines()[:5]:
    line.set_color("tab:blue")
    line.set_linewidth(2)
for line in axs[3].get_lines()[10:15]:
    line.set_color("tab:orange")
    line.set_linewidth(2)
for line in axs[3].get_lines()[15:20]:
    line.set_color("tab:green")
    line.set_linewidth(2)
for line in axs[3].get_lines()[20:25]:
    line.set_color("tab:red")
    line.set_linewidth(2)
for line in axs[3].get_lines()[25:30]:
    line.set_color("#bcbd22")
    line.set_linewidth(2)
for line in axs[3].get_lines()[30:35]:
    line.set_color("#17becf")
    line.set_linewidth(2)
for line in axs[3].get_lines()[40:45]:
    line.set_color("tab:purple")
    line.set_linewidth(2)
for line in axs[3].get_lines()[45:50]:
    line.set_color("#7f7f7f")
    line.set_linewidth(2)
for line in axs[3].get_lines()[50:55]:
    line.set_color("tab:brown")
    line.set_linewidth(2)
for line in axs[3].get_lines()[55:60]:
    line.set_color("tab:pink")
    line.set_linewidth(2)
for line in axs[3].get_lines()[60:65]:
    line.set_color("blue")
    line.set_linewidth(2)

fig.tight_layout()
plt.savefig("box_plot.pdf")
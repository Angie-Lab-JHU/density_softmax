import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import pickle
import robustness_metrics as rm
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# corruption_types = [
# 	'gaussian_noise',
#     'shot_noise',
#     'impulse_noise',
#     'defocus_blur',
#     'frosted_glass_blur',
#     'motion_blur',
#     'zoom_blur',
#     'snow',
#     'frost',
#     'fog',
#     'brightness',
#     'contrast',
#     'elastic',
#     'pixelate',
#     'jpeg_compression',
# ]

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
	# miss_probs, miss_labels = [], []
	# idx = 0
	# for prob in probs:
	# 	max_value = np.argmax(prob)
	# 	if max_value != labels[idx]:
	# 		miss_labels.append(labels[idx])
	# 		miss_probs.append(prob)
	# 	idx += 1
	# metrics['test/ece'].add_batch(miss_probs, label=miss_labels)
	return metrics['test/negative_log_likelihood'].result().numpy(), metrics['test/accuracy'].result().numpy() * 100, metrics['test/ece'].result()['ece']
	

def create_skewgraph(method):
	nll, acc, ece = [], [], []
	for level in range(1,6):
		for corruption_type in corruption_types:
			nll_tmp, acc_tmp, ece_tmp = get_results(method, f'{corruption_type}_{level}')
			nll.append(nll_tmp)
			acc.append(acc_tmp)
			ece.append(ece_tmp)
			for metric in metrics.values():
				metric.reset_states()
		
	return nll, acc, ece 


nll_erm, acc_erm, ece_erm = get_results("deepensembles", "clean")
for metric in metrics.values():
	metric.reset_states()

nll_erm_ood, acc_erm_ood, ece_erm_ood = create_skewgraph("deepensembles")
print(nll_erm)
print(acc_erm)
print(ece_erm)
print(np.mean(nll_erm_ood))
print(np.mean(acc_erm_ood))
print(np.mean(ece_erm_ood))



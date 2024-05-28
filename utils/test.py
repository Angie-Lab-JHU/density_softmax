import os
import pickle
import tensorflow as tf
import robustness_metrics as rm

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# corruption_types = [
# 		'gaussian_noise',
# 		'shot_noise',
# 		'impulse_noise',
# 		'defocus_blur',
# 		'frosted_glass_blur',
# 		'motion_blur',
# 		'zoom_blur',
# 		'snow',
# 		'frost',
# 		'fog',
# 		'brightness',
# 		'contrast',
# 		'elastic',
# 		'pixelate',
# 		'jpeg_compression',
# 	]

# def compare(data_name, ds_name, erm_name):
# 	metrics = {
# 		'ds': tf.keras.metrics.SparseCategoricalAccuracy(),
# 		'erm': tf.keras.metrics.SparseCategoricalAccuracy()
# 	}

# 	with open("out/density_softmax/cifar10/model" + str(ds_name) + "/" + data_name + "_probs.pkl", "rb") as fp:
# 		ds_probs = pickle.load(fp)

# 	with open("out/density_softmax/cifar10/model" + str(ds_name) + "/" + data_name + "_labels.pkl", "rb") as fp:
# 		ds_labels = pickle.load(fp)

# 	with open("out/deterministic/cifar10/model" + str(erm_name) + "/" + data_name + "_probs.pkl", "rb") as fp:
# 		erm_probs = pickle.load(fp)

# 	with open("out/deterministic/cifar10/model" + str(erm_name) + "/" + data_name + "_labels.pkl", "rb") as fp:
# 		erm_labels = pickle.load(fp)

# 	metrics['ds'].update_state(ds_labels, ds_probs)
# 	metrics['erm'].update_state(erm_labels, erm_probs)

# 	ds_acc = metrics['ds'].result().numpy() * 100
# 	erm_acc = metrics['erm'].result().numpy() * 100

# 	diff_acc = ds_acc - erm_acc
# 	data_key = data_name + str(ds_name) + str(erm_name)
# 	return diff_acc, data_key, ds_acc

# acc_dict = {}

# for ds_name in range(1, 10):
# 	for erm_name in range(1, 10):
# 		for level in range(1, 6):
# 			for corruption_type in corruption_types:
# 				diff_acc, data_name, ds_acc = compare(f'{corruption_type}_{level}', ds_name, erm_name)
# 				if ds_acc > 90:
# 					acc_dict[data_name] = diff_acc

# acc_dict = dict(sorted(acc_dict.items(), key=lambda item: item[1]))
# print(acc_dict)
# quit()

metrics = {'ds': tf.keras.metrics.SparseCategoricalAccuracy()}

with open("out/mimo/cifar10/model1/clean_probs.pkl", "rb") as fp:
	ds_probs = pickle.load(fp)

with open("out/mimo/cifar10/model1/clean_labels.pkl", "rb") as fp:
	ds_labels = pickle.load(fp)

metrics['ds'].update_state(ds_labels, ds_probs)
ds_acc = metrics['ds'].result().numpy() * 100
print(ds_acc)
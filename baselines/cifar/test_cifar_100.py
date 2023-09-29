# coding=utf-8
# Copyright 2023 The Uncertainty Baselines Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Wide ResNet 28-10 on CIFAR-10/100 trained with maximum likelihood.

Hyperparameters differ slightly from the original paper's code
(https://github.com/szagoruyko/wide-residual-networks) as TensorFlow uses, for
example, l2 instead of weight decay, and a different parameterization for SGD's
momentum.
"""

import os
import time
from absl import app
from absl import flags
from absl import logging
import robustness_metrics as rm
import tensorflow as tf
import tensorflow_datasets as tfds
import uncertainty_baselines as ub
import ood_utils  # local file import from baselines.cifar
import utils  # local file import from baselines.cifar
from tensorboard.plugins.hparams import api as hp
import numpy as np
import tensorflow_probability as tfp
import pickle
from tensorflow import keras
from tensorflow.keras import regularizers
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

flags.DEFINE_float('label_smoothing', 0., 'Label smoothing parameter in [0,1].')
flags.register_validator('label_smoothing',
												 lambda ls: ls >= 0.0 and ls <= 1.0,
												 message='--label_smoothing must be in [0, 1].')

# Data Augmentation flags.
flags.DEFINE_bool('augmix', False,
									'Whether to perform AugMix [4] on the input data.')
flags.DEFINE_integer('aug_count', 1,
										 'Number of augmentation operations in AugMix to perform '
										 'on the input image. In the simgle model context, it'
										 'should be 1. In the ensembles context, it should be'
										 'ensemble_size if we perform random_augment only; It'
										 'should be (ensemble_size - 1) if we perform augmix.')
flags.DEFINE_float('augmix_prob_coeff', 0.5, 'Augmix probability coefficient.')
flags.DEFINE_integer('augmix_depth', -1,
										 'Augmix depth, -1 meaning sampled depth. This corresponds'
										 'to line 7 in the Algorithm box in [4].')
flags.DEFINE_integer('augmix_width', 3,
										 'Augmix width. This corresponds to the k in line 5 in the'
										 'Algorithm box in [4].')

# Fine-grained specification of the hyperparameters (used when FLAGS.l2 is None)
flags.DEFINE_float('bn_l2', None, 'L2 reg. coefficient for batch-norm layers.')
flags.DEFINE_float('input_conv_l2', None,
									 'L2 reg. coefficient for the input conv layer.')
flags.DEFINE_float('group_1_conv_l2', None,
									 'L2 reg. coefficient for the 1st group of conv layers.')
flags.DEFINE_float('group_2_conv_l2', None,
									 'L2 reg. coefficient for the 2nd group of conv layers.')
flags.DEFINE_float('group_3_conv_l2', None,
									 'L2 reg. coefficient for the 3rd group of conv layers.')
flags.DEFINE_float('dense_kernel_l2', None,
									 'L2 reg. coefficient for the kernel of the dense layer.')
flags.DEFINE_float('dense_bias_l2', None,
									 'L2 reg. coefficient for the bias of the dense layer.')


flags.DEFINE_bool('collect_profile', False,
									'Whether to trace a profile with tensorboard')

# OOD flags.
flags.DEFINE_bool('eval_only', True,
									'Whether to run only eval and (maybe) OOD steps.')
flags.DEFINE_bool('eval_on_ood', True,
									'Whether to run OOD evaluation on specified OOD datasets.')
flags.DEFINE_list('ood_dataset', 'cifar10,svhn_cropped',
									'list of OOD datasets to evaluate on.')
flags.DEFINE_string('saved_model_dir', None,
										'Directory containing the saved model checkpoints.')
flags.DEFINE_bool('dempster_shafer_ood', False,
									'Wheter to use DempsterShafer Uncertainty score.')


flags.DEFINE_string('corruption_type', None, 'corruption_type')
flags.DEFINE_integer('severity', None, 'Number')

FLAGS = flags.FLAGS
# FLAGS.dataset = "cifar10"
FLAGS.dataset = "cifar100"

def _extract_hyperparameter_dictionary():
	"""Create the dictionary of hyperparameters from FLAGS."""
	flags_as_dict = FLAGS.flag_values_dict()
	hp_keys = ub.models.get_wide_resnet_hp_keys()
	hps = {k: flags_as_dict[k] for k in hp_keys}
	return hps

def Coupling(input_shape):
	output_dim = 16
	reg = 0.01
	input = keras.layers.Input(shape=input_shape)

	t_layer_1 = keras.layers.Dense(
		output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg)
	)(input)
	t_layer_2 = keras.layers.Dense(
		output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg)
	)(t_layer_1)
	t_layer_3 = keras.layers.Dense(
		output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg)
	)(t_layer_2)
	t_layer_4 = keras.layers.Dense(
		output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg)
	)(t_layer_3)
	t_layer_5 = keras.layers.Dense(
		input_shape, activation="linear", kernel_regularizer=regularizers.l2(reg)
	)(t_layer_4)

	s_layer_1 = keras.layers.Dense(
		output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg)
	)(input)
	s_layer_2 = keras.layers.Dense(
		output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg)
	)(s_layer_1)
	s_layer_3 = keras.layers.Dense(
		output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg)
	)(s_layer_2)
	s_layer_4 = keras.layers.Dense(
		output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg)
	)(s_layer_3)
	s_layer_5 = keras.layers.Dense(
		input_shape, activation="tanh", kernel_regularizer=regularizers.l2(reg)
	)(s_layer_4)

	return keras.Model(inputs=input, outputs=[s_layer_5, t_layer_5])

class RealNVP(keras.Model):
	def __init__(self, num_coupling_layers, input_dim):
		super(RealNVP, self).__init__()

		self.num_coupling_layers = num_coupling_layers

		# Distribution of the latent space.  tf.int32)
		self.distribution = tfp.distributions.MultivariateNormalDiag(
			loc=np.zeros(input_dim, dtype="float32"), scale_diag=np.ones(input_dim, dtype="float32")
		)
		# self.masks = np.array(
		#     [[0, 1], [1, 0]] * (num_coupling_layers // 2), dtype="float32"
		# )
		self.masks = np.array(
			[np.concatenate((np.zeros(input_dim // 2), np.ones(input_dim // 2))), np.concatenate((np.ones(input_dim // 2), np.zeros(input_dim // 2)))] * (num_coupling_layers // 2), dtype="float32"
		)

		self.loss_tracker = keras.metrics.Mean(name="loss")
		self.layers_list = [Coupling(input_dim) for i in range(num_coupling_layers)]

	@property
	def metrics(self):
		"""List of the model's metrics.

		We make sure the loss tracker is listed as part of `model.metrics`
		so that `fit()` and `evaluate()` are able to `reset()` the loss tracker
		at the start of each epoch and at the start of an `evaluate()` call.
		"""
		return [self.loss_tracker]

	def call(self, x, training=True):
		log_det_inv = 0
		direction = 1
		if training:
			direction = -1
		for i in range(self.num_coupling_layers)[::direction]:
			x_masked = x * self.masks[i]
			reversed_mask = 1 - self.masks[i]
			s, t = self.layers_list[i](x_masked)
			s *= reversed_mask
			t *= reversed_mask
			gate = (direction - 1) / 2
			x = (
				reversed_mask
				* (x * tf.exp(direction * s) + direction * t * tf.exp(gate * s))
				+ x_masked
			)
			log_det_inv += gate * tf.reduce_sum(s, [1])

		return x, log_det_inv

	# Log likelihood of the normal distribution plus the log determinant of the jacobian.

	def log_loss(self, x):
		y, logdet = self(x)
		log_likelihood = self.distribution.log_prob(y) + logdet
		return -tf.reduce_mean(log_likelihood)

	def score_samples(self, x):
		y, logdet = self(x)
		log_likelihood = self.distribution.log_prob(y) + logdet
		return log_likelihood

#Cifar-10 -0.55 -0.45
#Cifar-100 -0.4 -0.3
def rescale(X, x_min, x_max, y_min = -0.4, y_max = -0.3):
# def rescale(X, x_min, x_max, y_min = -0.55, y_max = -0.45):
  X = (X - x_min) / (x_max - x_min)
  X = X * (y_max - y_min)
  X = X + y_min
  return tf.cast(X, tf.float64)

def main(argv):
	fmt = '[%(filename)s:%(lineno)s] %(message)s'
	formatter = logging.PythonFormatter(fmt)
	logging.get_absl_handler().setFormatter(formatter)
	del argv  # unused arg

	tf.io.gfile.makedirs(FLAGS.output_dir)
	logging.info('Saving checkpoints at %s', FLAGS.output_dir)
	tf.random.set_seed(FLAGS.seed)

	data_dir = FLAGS.data_dir
	if FLAGS.use_gpu:
		logging.info('Use GPU')
		strategy = tf.distribute.MirroredStrategy()

	aug_params = {
			'augmix': FLAGS.augmix,
			'aug_count': FLAGS.aug_count,
			'augmix_depth': FLAGS.augmix_depth,
			'augmix_prob_coeff': FLAGS.augmix_prob_coeff,
			'augmix_width': FLAGS.augmix_width,
	}
	batch_size = FLAGS.per_core_batch_size * FLAGS.num_cores
	seeds = tf.random.experimental.stateless_split(
			[FLAGS.seed, FLAGS.seed + 1], 2)[:, 0]
	train_builder = ub.datasets.get(
			FLAGS.dataset,
			data_dir=data_dir,
			download_data=True,
			split=tfds.Split.TRAIN,
			seed=seeds[0],
			aug_params=aug_params,
			shuffle_buffer_size=FLAGS.shuffle_buffer_size,
			validation_percent=1. - FLAGS.train_proportion,
	)
	train_dataset = train_builder.load(batch_size=batch_size)

	clean_test_builder = ub.datasets.get(
	  FLAGS.dataset,
			split=tfds.Split.TEST,
			data_dir=data_dir,
			drop_remainder=FLAGS.drop_remainder_for_eval)
	clean_test_dataset = clean_test_builder.load(batch_size=batch_size)
	corruption_types, _ = utils.load_corrupted_test_info(FLAGS.dataset)
	datasets_to_evaluate = {
	# 	'clean': strategy.experimental_distribute_dataset(clean_test_dataset),
	}
	steps_per_eval = clean_test_builder.num_examples // batch_size
	# steps_per_evals = [clean_test_builder.num_examples // batch_size]
	test_datasets = {
			# 'clean': strategy.experimental_distribute_dataset(clean_test_dataset),
	}
	data_dir = data_dir + "/cifar100_corrupted"
	# steps_per_evals = [2021 // batch_size, 2000 // batch_size]
	# corruption_types = ["v4", "v6"]
	# for corruption_type in corruption_types:
	# 	dataset = ub.datasets.get(
	# 					f'{FLAGS.dataset}_1',
	# 					corruption_type=corruption_type,
	# 					download_data=True,
	# 					split=tfds.Split.TEST,
	# 					data_dir=data_dir).load(batch_size=batch_size)
	# 	datasets_to_evaluate[f'{corruption_type}'] = (
	# 					strategy.experimental_distribute_dataset(dataset))

	# dataset = ub.datasets.get(
	# 					f'{FLAGS.dataset}_corrupted',
	# 					corruption_type=FLAGS.corruption_type,
	# 					data_dir=data_dir,
	# 					severity=FLAGS.severity,
	# 					split=tfds.Split.TEST).load(
	# 							batch_size=batch_size)
	# test_datasets[f'{FLAGS.corruption_type}_{FLAGS.severity}'] = (
	# 					strategy.experimental_distribute_dataset(dataset))
	
	for corruption_type in corruption_types:
		for severity in range(1, 6):
			dataset = ub.datasets.get(
						f'{FLAGS.dataset}_corrupted',
						download_data = True,
						corruption_type=corruption_type,
						severity=severity,
						split=tfds.Split.TEST,
						data_dir=data_dir).load(batch_size=batch_size)
			datasets_to_evaluate[f'{corruption_type}_{severity}'] = (
						strategy.experimental_distribute_dataset(dataset))

	seeds = tf.random.experimental.stateless_split(
			[FLAGS.seed, FLAGS.seed + 1], 2)[:, 0]

	ds_info = tfds.builder(FLAGS.dataset).info
	train_dataset_size = (
			ds_info.splits['train'].num_examples * FLAGS.train_proportion)
	steps_per_epoch = int(train_dataset_size / batch_size)
	with strategy.scope():
		logging.info('Building ResNet model')
		model = ub.models.wide_resnet(
				input_shape=(32, 32, 3),
				depth=28,
				width_multiplier=10,
				num_classes=100,
				l2=FLAGS.l2,
				hps=_extract_hyperparameter_dictionary(),
				seed=seeds[1])
		density_model = RealNVP(num_coupling_layers=2, input_dim = 640)
		density_model.build(input_shape=(None,640))

		encoder = tf.keras.Model(model.input, model.get_layer('flatten').output)
		classifer = tf.keras.Model(model.get_layer('dense').input, model.get_layer('dense').output)
		metrics = {
				'test/negative_log_likelihood':
						tf.keras.metrics.Mean(),
				'test/accuracy':
						tf.keras.metrics.SparseCategoricalAccuracy(),
				'test/ece':
						rm.metrics.ExpectedCalibrationError(num_bins=FLAGS.num_bins),
		}
		base_lr = FLAGS.base_learning_rate * batch_size / 128
		lr_decay_epochs = [(int(start_epoch_str) * FLAGS.train_epochs) // 200
											 for start_epoch_str in FLAGS.lr_decay_epochs]
		lr_schedule = ub.schedules.WarmUpPiecewiseConstantSchedule(
				steps_per_epoch,
				base_lr,
				decay_ratio=FLAGS.lr_decay_ratio,
				decay_epochs=lr_decay_epochs,
				warmup_epochs=FLAGS.lr_warmup_epochs)
		optimizer = tf.keras.optimizers.SGD(lr_schedule, momentum=1.0 - FLAGS.one_minus_momentum, nesterov=True)
		checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
		latest_checkpoint = tf.train.latest_checkpoint(FLAGS.output_dir)
		latest_checkpoint = latest_checkpoint.replace("density_model", "checkpoint-10")
		checkpoint.restore(latest_checkpoint)

	@tf.function
	def test_density(iterator, num_steps):
		"""Evaluation StepFn."""
		def step_fn(inputs):
			"""Per-Replica StepFn."""
			images = inputs['features']
			latents = encoder(images, training=False)
			return density_model.score_samples(latents)

		nll = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
		for _ in tf.range(tf.cast(num_steps, tf.int32)):
			out = strategy.run(step_fn, args=(next(iterator),))
			nll = nll.write(nll.size(), out)
			
		return nll.stack()

	@tf.function
	def save_2_plot(iterator, num_steps):
		"""Evaluation StepFn."""
		def step_fn(inputs):
			images = inputs['features']
			labels = inputs['labels']

			latents = encoder(images, training=False)
			log_likelihood = density_model.score_samples(latents)
			log_likelihood = rescale(log_likelihood, x_min, x_max)
			prob = tf.math.exp(log_likelihood)
			prob = tf.expand_dims(prob, 1)
			logits = classifer(latents)
			probs = tf.nn.softmax(logits * tf.cast(prob, dtype=tf.float32))
			return probs, labels

		probs = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
		labels = tf.TensorArray(tf.int32, size=0, dynamic_size=True)
		# labels = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
		for _ in tf.range(tf.cast(num_steps, tf.int32)):
			prob, label = strategy.run(step_fn, args=(next(iterator),))
			probs = probs.write(probs.size(), prob)
			labels = labels.write(labels.size(), label)
		
		return probs.stack(), labels.stack()

	@tf.function
	def test_step(iterator, dataset_split, dataset_name, num_steps, x_max, x_min):
		"""Evaluation StepFn."""
		def step_fn(inputs):
			"""Per-Replica StepFn."""
			images = inputs['features']
			labels = inputs['labels']

			latents = encoder(images, training=False)
			log_likelihood = density_model.score_samples(latents)
			log_likelihood = rescale(log_likelihood, x_min, x_max)
			prob = tf.math.exp(log_likelihood)
			prob = tf.expand_dims(prob, 1)
			logits = classifer(latents)
			probs = tf.nn.softmax(logits * tf.cast(prob, dtype=tf.float32))

			negative_log_likelihood = tf.reduce_mean(
					tf.keras.losses.sparse_categorical_crossentropy(labels, probs))

			metrics['test/negative_log_likelihood'].update_state(
						negative_log_likelihood)
			metrics['test/accuracy'].update_state(labels, probs)
			metrics['test/ece'].add_batch(probs, label=labels)

		for _ in tf.range(tf.cast(num_steps, tf.int32)):
			strategy.run(step_fn, args=(next(iterator),))
	
	train_iterator = iter(train_dataset)
	density_model.load_weights('checkpoints/density_softmax/cifar100/model1/density_model')
	train_nll = test_density(train_iterator, steps_per_epoch)
	train_nll = tf.reshape(train_nll,[-1])
	x_max = tf.reduce_max(train_nll)
	x_min = tf.reduce_min(train_nll)

	# ood_type = "clean"
	# ood_type = f'{FLAGS.corruption_type}_{FLAGS.severity}'
	# datasets_to_evaluate = {ood_type: test_datasets[ood_type]}
	for dataset_name, test_dataset in datasets_to_evaluate.items():
		test_iterator = iter(test_dataset)
		probs, labels = save_2_plot(test_iterator, steps_per_eval)
		probs = probs.numpy()
		probs= probs.reshape((probs.shape[0] * probs.shape[1]), probs.shape[2])
		labels = labels.numpy()
		with open("out/tmp_cifar100/density_softmax/" + dataset_name + "_probs.pkl", "wb") as fp:
			pickle.dump(probs, fp)
		with open("out/tmp_cifar100/density_softmax/" + dataset_name + "_labels.pkl", "wb") as fp:
			pickle.dump(labels.ravel(), fp)
	
	quit()
	cAcc, cECE, cNLL = [], [], []
	idx = 0
	for dataset_name, test_dataset in datasets_to_evaluate.items():
		test_iterator = iter(test_dataset)
		logging.info('Testing on dataset %s', dataset_name)

		test_start_time = time.time()
		test_step(test_iterator, 'test', dataset_name, steps_per_evals[idx], x_max, x_min)
		ms_per_example = (time.time() - test_start_time) * 1e6 / batch_size
		logging.info('Test NLL: %.4f, Accuracy: %.2f%%, ECE: %.4f',
								 metrics['test/negative_log_likelihood'].result(),
								 metrics['test/accuracy'].result() * 100,
								 metrics['test/ece'].result()['ece'],)
		if dataset_name != 'clean':
			cNLL.append(metrics['test/negative_log_likelihood'].result())
			cAcc.append(metrics['test/accuracy'].result() * 100)
			cECE.append(metrics['test/ece'].result()['ece'])

		for metric in metrics.values():
			metric.reset_states()
		
		idx += 1

	cNLL = np.asarray(cNLL)
	cAcc = np.asarray(cAcc)
	cECE = np.asarray(cECE)
	print(np.mean(cNLL))
	print(np.mean(cAcc))
	print(np.mean(cECE))

if __name__ == '__main__':
	app.run(main)
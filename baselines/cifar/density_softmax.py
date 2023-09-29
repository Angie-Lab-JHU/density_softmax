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

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
flags.DEFINE_bool('eval_only', False,
									'Whether to run only eval and (maybe) OOD steps.')
flags.DEFINE_bool('eval_on_ood', False,
									'Whether to run OOD evaluation on specified OOD datasets.')
flags.DEFINE_list('ood_dataset', 'cifar100,svhn_cropped',
									'list of OOD datasets to evaluate on.')
flags.DEFINE_string('saved_model_dir', None,
										'Directory containing the saved model checkpoints.')
flags.DEFINE_bool('dempster_shafer_ood', False,
									'Wheter to use DempsterShafer Uncertainty score.')


FLAGS = flags.FLAGS
# FLAGS.dataset = "cifar100"
# FLAGS.download_data = True

def _extract_hyperparameter_dictionary():
	"""Create the dictionary of hyperparameters from FLAGS."""
	flags_as_dict = FLAGS.flag_values_dict()
	hp_keys = ub.models.get_wide_resnet_hp_keys()
	hps = {k: flags_as_dict[k] for k in hp_keys}
	return hps


def _generalized_energy_distance(labels, predictions, num_classes):
	"""Compute generalized energy distance.

	See Eq. (8) https://arxiv.org/abs/2006.06015
	where d(a, b) = (a - b)^2.

	Args:
		labels: [batch_size, num_classes] Tensor with empirical probabilities of
			each class assigned by the labellers.
		predictions: [batch_size, num_classes] Tensor of predicted probabilities.
		num_classes: Integer.

	Returns:
		Tuple of Tensors (label_diversity, sample_diversity, ged).
	"""
	y = tf.expand_dims(labels, -1)
	y_hat = tf.expand_dims(predictions, -1)

	non_diag = tf.expand_dims(1.0 - tf.eye(num_classes), 0)
	distance = tf.reduce_sum(tf.reduce_sum(
			non_diag * y * tf.transpose(y_hat, perm=[0, 2, 1]), -1), -1)
	label_diversity = tf.reduce_sum(tf.reduce_sum(
			non_diag * y * tf.transpose(y, perm=[0, 2, 1]), -1), -1)
	sample_diversity = tf.reduce_sum(tf.reduce_sum(
			non_diag * y_hat * tf.transpose(y_hat, perm=[0, 2, 1]), -1), -1)
	ged = tf.reduce_mean(2 * distance - label_diversity - sample_diversity)
	return label_diversity, sample_diversity, ged


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

# def rescale(X, x_min, x_max, y_min = -0.4, y_max = -0.3):
def rescale(X, x_min, x_max, y_min = -0.55, y_max = -0.45):
  X = (X - x_min) / (x_max - x_min)
  X = X * (y_max - y_min)
  X = X + y_min
  return tf.cast(X, tf.float64)

def rescale_normal(X, x_min, x_max, y_min = -1, y_max = 0):
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
	else:
		logging.info('Use TPU at %s',
								 FLAGS.tpu if FLAGS.tpu is not None else 'local')
		resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=FLAGS.tpu)
		tf.config.experimental_connect_to_cluster(resolver)
		tf.tpu.experimental.initialize_tpu_system(resolver)
		strategy = tf.distribute.TPUStrategy(resolver)

	ds_info = tfds.builder(FLAGS.dataset).info
	batch_size = FLAGS.per_core_batch_size * FLAGS.num_cores
	train_dataset_size = (
			ds_info.splits['train'].num_examples * FLAGS.train_proportion)
	steps_per_epoch = int(train_dataset_size / batch_size)
	logging.info('Steps per epoch %s', steps_per_epoch)
	logging.info('Size of the dataset %s', ds_info.splits['train'].num_examples)
	logging.info('Train proportion %s', FLAGS.train_proportion)
	steps_per_eval = ds_info.splits['test'].num_examples // batch_size
	num_classes = ds_info.features['label'].num_classes

	aug_params = {
			'augmix': FLAGS.augmix,
			'aug_count': FLAGS.aug_count,
			'augmix_depth': FLAGS.augmix_depth,
			'augmix_prob_coeff': FLAGS.augmix_prob_coeff,
			'augmix_width': FLAGS.augmix_width,
	}

	# Note that stateless_{fold_in,split} may incur a performance cost, but a
	# quick side-by-side test seemed to imply this was minimal.
	seeds = tf.random.experimental.stateless_split(
			[FLAGS.seed, FLAGS.seed + 1], 2)[:, 0]
	train_builder = ub.datasets.get(
			FLAGS.dataset,
			data_dir=data_dir,
			download_data=FLAGS.download_data,
			split=tfds.Split.TRAIN,
			seed=seeds[0],
			aug_params=aug_params,
			shuffle_buffer_size=FLAGS.shuffle_buffer_size,
			validation_percent=1. - FLAGS.train_proportion,
	)
	train_dataset = train_builder.load(batch_size=batch_size)
	validation_dataset = None
	steps_per_validation = 0
	if FLAGS.train_proportion < 1.0:
		validation_builder = ub.datasets.get(
				FLAGS.dataset,
				split=tfds.Split.VALIDATION,
				validation_percent=1. - FLAGS.train_proportion,
				data_dir=data_dir,
				drop_remainder=FLAGS.drop_remainder_for_eval)
		validation_dataset = validation_builder.load(batch_size=batch_size)
		validation_dataset = strategy.experimental_distribute_dataset(
				validation_dataset)
		steps_per_validation = validation_builder.num_examples // batch_size
	clean_test_builder = ub.datasets.get(
	  FLAGS.dataset,
			split=tfds.Split.TEST,
      data_dir=data_dir,
      drop_remainder=FLAGS.drop_remainder_for_eval)
	clean_test_dataset = clean_test_builder.load(batch_size=batch_size)
	test_datasets = {
			'clean': strategy.experimental_distribute_dataset(clean_test_dataset),
	}


	train_dataset = strategy.experimental_distribute_dataset(train_dataset)

	steps_per_epoch = train_builder.num_examples // batch_size
	steps_per_eval = clean_test_builder.num_examples // batch_size
	num_classes = 100 if FLAGS.dataset == 'cifar100' else 10

	if FLAGS.eval_on_ood:
		ood_dataset_names = FLAGS.ood_dataset
		ood_ds, steps_per_ood = ood_utils.load_ood_datasets(
				ood_dataset_names,
				clean_test_builder,
				1. - FLAGS.train_proportion,
				batch_size,
				drop_remainder=FLAGS.drop_remainder_for_eval)
		ood_datasets = {
				name: strategy.experimental_distribute_dataset(ds)
				for name, ds in ood_ds.items()
		}

	if FLAGS.corruptions_interval > 0:
		if FLAGS.dataset == 'cifar100':
			data_dir = FLAGS.cifar100_c_path
		corruption_types, _ = utils.load_corrupted_test_info(FLAGS.dataset)
		for corruption_type in corruption_types:
			for severity in range(1, 6):
				dataset = ub.datasets.get(
						f'{FLAGS.dataset}_corrupted',
						corruption_type=corruption_type,
						severity=severity,
						split=tfds.Split.TEST,
						data_dir=data_dir).load(batch_size=batch_size)
				test_datasets[f'{corruption_type}_{severity}'] = (
						strategy.experimental_distribute_dataset(dataset))

	summary_writer = tf.summary.create_file_writer(
			os.path.join(FLAGS.output_dir, 'summaries'))

	with strategy.scope():
		logging.info('Building ResNet model')
		model = ub.models.wide_resnet(
				input_shape=(32, 32, 3),
				depth=28,
				width_multiplier=10,
				num_classes=num_classes,
				l2=FLAGS.l2,
				hps=_extract_hyperparameter_dictionary(),
				seed=seeds[1])
		
		density_model = RealNVP(num_coupling_layers=2, input_dim = 640)
		density_model.build(input_shape=(None,640))
		
		encoder = tf.keras.Model(model.input, model.get_layer('flatten').output)
		classifer = tf.keras.Model(model.get_layer('dense').input, model.get_layer('dense').output)
		logging.info('Model input shape: %s', model.input_shape)
		logging.info('Model output shape: %s', model.output_shape)
		logging.info('Model number of weights: %s', model.count_params())
		# Linearly scale learning rate and the decay epochs by vanilla settings.
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
		density_optimizer = tf.keras.optimizers.Adam(1e-4)
		metrics = {
				'train/negative_log_likelihood':
						tf.keras.metrics.Mean(),
				'train/accuracy':
						tf.keras.metrics.SparseCategoricalAccuracy(),
				'train/loss':
						tf.keras.metrics.Mean(),
				'train/density_loss':
						tf.keras.metrics.Mean(),
				'train/ece':
						rm.metrics.ExpectedCalibrationError(num_bins=FLAGS.num_bins),
				'normalize/accuracy':
						tf.keras.metrics.SparseCategoricalAccuracy(),
				'normalize/loss':
						tf.keras.metrics.Mean(),
				'test/negative_log_likelihood':
						tf.keras.metrics.Mean(),
				'test/accuracy':
						tf.keras.metrics.SparseCategoricalAccuracy(),
				'test/ece':
						rm.metrics.ExpectedCalibrationError(num_bins=FLAGS.num_bins),
				'test_ood/negative_log_likelihood':
						tf.keras.metrics.Mean(),
				'test_ood/accuracy':
						tf.keras.metrics.SparseCategoricalAccuracy(),
				'test_ood/ece':
						rm.metrics.ExpectedCalibrationError(num_bins=FLAGS.num_bins),
		}
		if validation_dataset:
			metrics.update({
					'validation/negative_log_likelihood': tf.keras.metrics.Mean(),
					'validation/accuracy': tf.keras.metrics.SparseCategoricalAccuracy(),
					'validation/ece': rm.metrics.ExpectedCalibrationError(
							num_bins=FLAGS.num_bins),
			})
		if FLAGS.eval_on_ood:
			ood_metrics = ood_utils.create_ood_metrics(ood_dataset_names)
			metrics.update(ood_metrics)
		if FLAGS.corruptions_interval > 0:
			corrupt_metrics = {}
			for intensity in range(1, 6):
				for corruption in corruption_types:
					dataset_name = '{0}_{1}'.format(corruption, intensity)
					corrupt_metrics['test/nll_{}'.format(dataset_name)] = (
							tf.keras.metrics.Mean())
					corrupt_metrics['test/accuracy_{}'.format(dataset_name)] = (
							tf.keras.metrics.SparseCategoricalAccuracy())
					corrupt_metrics['test/ece_{}'.format(dataset_name)] = (
							rm.metrics.ExpectedCalibrationError(num_bins=FLAGS.num_bins))

		checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
		latest_checkpoint = tf.train.latest_checkpoint(FLAGS.output_dir)
		initial_epoch = 0
		if latest_checkpoint:
			# checkpoint.restore must be within a strategy.scope() so that optimizer
			# slot variables are mirrored.
			checkpoint.restore(latest_checkpoint)
			logging.info('Loaded checkpoint %s', latest_checkpoint)
			initial_epoch = optimizer.iterations.numpy() // steps_per_epoch

		if FLAGS.saved_model_dir:
			logging.info('Saved model dir : %s', FLAGS.saved_model_dir)
			latest_checkpoint = tf.train.latest_checkpoint(FLAGS.saved_model_dir)
			checkpoint.restore(latest_checkpoint)
			logging.info('Loaded checkpoint %s', latest_checkpoint)
		if FLAGS.eval_only:
			initial_epoch = FLAGS.train_epochs - 1  # Run just one epoch of eval

	@tf.function
	def train_step(iterator):
		"""Training StepFn."""
		def step_fn(inputs):
			"""Per-Replica StepFn."""
			images = inputs['features']
			labels = inputs['labels']

			if FLAGS.augmix and FLAGS.aug_count >= 1:
				# Index 0 at augmix processing is the unperturbed image.
				# We take just 1 augmented image from the returned augmented images.
				images = images[:, 1, ...]
			with tf.GradientTape(persistent=True) as tape:
				tape.watch(images)
				latents = encoder(images, training=True)
				logits = classifer(latents, training=True)
				if FLAGS.label_smoothing == 0.:
					negative_log_likelihood = tf.reduce_mean(
							tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True))
				else:
					one_hot_labels = tf.one_hot(tf.cast(labels, tf.int32), num_classes)
					negative_log_likelihood = tf.reduce_mean(
							tf.keras.losses.categorical_crossentropy(
									one_hot_labels,
									logits,
									from_logits=True,
									label_smoothing=FLAGS.label_smoothing))
				l2_loss = sum(model.losses)
				grad = tape.gradient(latents, images)
				grad_norm = tf.sqrt(tf.reduce_sum(grad ** 2, axis = [1, 2, 3]))
				jacob_loss = 1e-4 * tf.reduce_mean(((grad_norm - 1) ** 2))
				# CIFAR-100
				# jacob_loss = 1e-5 * tf.reduce_mean(((grad_norm - 1) ** 2))
				
				loss = negative_log_likelihood + l2_loss + jacob_loss
				# Scale the loss given the TPUStrategy will reduce sum all gradients.
				scaled_loss = loss / strategy.num_replicas_in_sync

			grads = tape.gradient(scaled_loss, model.trainable_variables)
			optimizer.apply_gradients(zip(grads, model.trainable_variables))

			probs = tf.nn.softmax(logits)
			metrics['train/ece'].add_batch(probs, label=labels)
			metrics['train/loss'].update_state(loss)
			metrics['train/negative_log_likelihood'].update_state(
					negative_log_likelihood)
			metrics['train/accuracy'].update_state(labels, logits)

		for _ in tf.range(tf.cast(steps_per_epoch, tf.int32)):
			strategy.run(step_fn, args=(next(iterator),))

	@tf.function
	def normalize_softmax(iterator):
		"""Training StepFn."""
		def step_fn(inputs):
			"""Per-Replica StepFn."""
			images = inputs['features']
			labels = inputs['labels']

			if FLAGS.augmix and FLAGS.aug_count >= 1:
				# Index 0 at augmix processing is the unperturbed image.
				# We take just 1 augmented image from the returned augmented images.
				images = images[:, 1, ...]
			latents = encoder(images, training=False)
			log_likelihood = density_model.score_samples(latents)
			log_likelihood = rescale(log_likelihood, x_min, x_max)
			prob = tf.math.exp(log_likelihood)
			prob = tf.expand_dims(prob, 1)

			with tf.GradientTape() as tape:
				logits = classifer(latents, training=True)
				probs = tf.nn.softmax(logits * tf.cast(prob, dtype=tf.float32))
				# probs = tf.nn.softmax(logits * tf.cast(1., dtype=tf.float32))

				if FLAGS.label_smoothing == 0.:
					negative_log_likelihood = tf.reduce_mean(
							tf.keras.losses.sparse_categorical_crossentropy(labels, probs, from_logits=False))
				else:
					one_hot_labels = tf.one_hot(tf.cast(labels, tf.int32), num_classes)
					negative_log_likelihood = tf.reduce_mean(
							tf.keras.losses.categorical_crossentropy(
									one_hot_labels,
									probs,
									from_logits=False,
									label_smoothing=FLAGS.label_smoothing))
				l2_loss = sum(classifer.losses)
				loss = negative_log_likelihood + l2_loss
				# Scale the loss given the TPUStrategy will reduce sum all gradients.
				scaled_loss = loss / strategy.num_replicas_in_sync

			grads = tape.gradient(scaled_loss, classifer.trainable_variables)
			optimizer.apply_gradients(zip(grads, classifer.trainable_variables))

			metrics['normalize/loss'].update_state(loss)
			metrics['normalize/accuracy'].update_state(labels, logits)

		for _ in tf.range(tf.cast(steps_per_epoch, tf.int32)):
			strategy.run(step_fn, args=(next(iterator),))

	def compute_loss(density_model, x):
		y, logdet = density_model(x)
		log_likelihood = density_model.distribution.log_prob(y) + logdet
		return -tf.reduce_mean(log_likelihood)

	@tf.function
	def train_density(iterator):
		"""Evaluation StepFn."""
		def step_fn(inputs):
			"""Per-Replica StepFn."""
			images = inputs['features']
			labels = inputs['labels']
			latents = encoder(images, training=False)
			with tf.GradientTape() as tape:
				loss = compute_loss(density_model, latents)
			
			metrics['train/density_loss'].update_state(loss)
			gradients = tape.gradient(loss, density_model.trainable_variables)
			density_optimizer.apply_gradients(zip(gradients, density_model.trainable_variables))
		for _ in tf.range(tf.cast(steps_per_epoch, tf.int32)):
			strategy.run(step_fn, args=(next(iterator),))

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
	def save_2_plot(iterator, num_steps, x_max, x_min):
		"""Evaluation StepFn."""
		def step_fn(inputs):
			"""Per-Replica StepFn."""
			images = inputs['features']
			labels = inputs['labels']
			latents = encoder(images, training=False)
			density_model.score_samples(latents)
			log_likelihood = density_model.score_samples(latents)
			log_likelihood = rescale(log_likelihood, x_min, x_max)
			prob = tf.math.exp(log_likelihood)
			prob = tf.expand_dims(prob, 1)

			logits = classifer(latents)
			probs = tf.nn.softmax(logits * tf.cast(prob, dtype=tf.float32))
			return probs, labels


		probs = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
		labels = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
		for _ in tf.range(tf.cast(num_steps, tf.int32)):
			prob, label = strategy.run(step_fn, args=(next(iterator),))
			probs = probs.write(probs.size(), prob)
			labels = labels.write(labels.size(), label)
		
		return probs.stack(), labels.stack()

	@tf.function
	def test_ood_step(iterator, dataset_name, num_steps, x_max, x_min):
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

			metrics['test_ood/negative_log_likelihood'].update_state(
						negative_log_likelihood)
			metrics['test_ood/accuracy'].update_state(labels, probs)
			metrics['test_ood/ece'].add_batch(probs, label=labels)

		for _ in tf.range(tf.cast(num_steps, tf.int32)):
			strategy.run(step_fn, args=(next(iterator),))
	
	@tf.function
	def test_step(iterator, dataset_split, dataset_name, num_steps):
		"""Evaluation StepFn."""
		def step_fn(inputs):
			"""Per-Replica StepFn."""
			images = inputs['features']
			labels = inputs['labels']
			logits = classifer(encoder(images))
			probs = tf.nn.softmax(logits)

			negative_log_likelihood = tf.reduce_mean(
					tf.keras.losses.sparse_categorical_crossentropy(labels, probs))

			if dataset_name == 'clean':
				metrics[f'{dataset_split}/negative_log_likelihood'].update_state(
						negative_log_likelihood)
				metrics[f'{dataset_split}/accuracy'].update_state(labels, probs)
				metrics[f'{dataset_split}/ece'].add_batch(probs, label=labels)
			elif dataset_name.startswith('ood/'):
				ood_labels = 1 - inputs['is_in_distribution']
				if FLAGS.dempster_shafer_ood:
					ood_scores = ood_utils.DempsterShaferUncertainty(logits)
				else:
					ood_scores = 1 - tf.reduce_max(probs, axis=-1)

				# Edgecase for if dataset_name contains underscores
				for name, metric in metrics.items():
					if dataset_name in name:
						metric.update_state(ood_labels, ood_scores)
			else:
				corrupt_metrics['test/nll_{}'.format(dataset_name)].update_state(
						negative_log_likelihood)
				corrupt_metrics['test/accuracy_{}'.format(dataset_name)].update_state(
						labels, probs)
				corrupt_metrics['test/ece_{}'.format(dataset_name)].add_batch(
						probs, label=labels)

		for _ in tf.range(tf.cast(num_steps, tf.int32)):
			strategy.run(step_fn, args=(next(iterator),))

	@tf.function
	def cifar10h_test_step(iterator, num_steps):
		"""Evaluation StepFn."""
		def step_fn(inputs):
			"""Per-Replica StepFn."""
			images = inputs['features']
			labels = inputs['labels']
			logits = model(images, training=False)

			negative_log_likelihood = tf.keras.losses.CategoricalCrossentropy(
					from_logits=True,
					reduction=tf.keras.losses.Reduction.NONE)(labels, logits)

			negative_log_likelihood = tf.reduce_mean(negative_log_likelihood)
			metrics['cifar10h/nll'].update_state(negative_log_likelihood)

			label_diversity, sample_diversity, ged = _generalized_energy_distance(
					labels, tf.nn.softmax(logits), 10)

			metrics['cifar10h/ged'].update_state(ged)
			metrics['cifar10h/ged_label_diversity'].update_state(
					tf.reduce_mean(label_diversity))
			metrics['cifar10h/ged_sample_diversity'].update_state(
					tf.reduce_mean(sample_diversity))

		for _ in tf.range(tf.cast(num_steps, tf.int32)):
			strategy.run(step_fn, args=(next(iterator),))

	metrics.update({'test/ms_per_example': tf.keras.metrics.Mean()})
	metrics.update({'train/ms_per_example': tf.keras.metrics.Mean()})

	train_iterator = iter(train_dataset)
	start_time = time.time()
	tb_callback = None
	if FLAGS.collect_profile:
		tb_callback = tf.keras.callbacks.TensorBoard(
				profile_batch=(100, 102),
				log_dir=os.path.join(FLAGS.output_dir, 'logs'))
		tb_callback.set_model(model)
	for epoch in range(initial_epoch, FLAGS.train_epochs):
		logging.info('Starting to run epoch: %s', epoch)
		if tb_callback:
			tb_callback.on_epoch_begin(epoch)
		if not FLAGS.eval_only:
			train_start_time = time.time()
			train_step(train_iterator)
			ms_per_example = (time.time() - train_start_time) * 1e6 / batch_size
			metrics['train/ms_per_example'].update_state(ms_per_example)

			current_step = (epoch + 1) * steps_per_epoch
			max_steps = steps_per_epoch * FLAGS.train_epochs
			time_elapsed = time.time() - start_time
			steps_per_sec = float(current_step) / time_elapsed
			eta_seconds = (max_steps - current_step) / steps_per_sec
			message = ('{:.1%} completion: epoch {:d}/{:d}. {:.1f} steps/s. '
								 'ETA: {:.0f} min. Time elapsed: {:.0f} min'.format(
										 current_step / max_steps, epoch + 1, FLAGS.train_epochs,
										 steps_per_sec, eta_seconds / 60, time_elapsed / 60))
			logging.info(message)
		if tb_callback:
			tb_callback.on_epoch_end(epoch)

		if validation_dataset:
			validation_iterator = iter(validation_dataset)
			test_step(
					validation_iterator, 'validation', 'clean', steps_per_validation)
		datasets_to_evaluate = {'clean': test_datasets['clean']}
		if (FLAGS.corruptions_interval > 0 and
				(epoch + 1) % FLAGS.corruptions_interval == 0):
			datasets_to_evaluate = test_datasets
		for dataset_name, test_dataset in datasets_to_evaluate.items():
			test_iterator = iter(test_dataset)
			logging.info('Testing on dataset %s', dataset_name)
			logging.info('Starting to run eval at epoch: %s', epoch)
			test_start_time = time.time()
			test_step(test_iterator, 'test', dataset_name, steps_per_eval)
			ms_per_example = (time.time() - test_start_time) * 1e6 / batch_size
			metrics['test/ms_per_example'].update_state(ms_per_example)

			logging.info('Done with testing on %s', dataset_name)

		if FLAGS.eval_on_ood:
			for ood_dataset_name, ood_dataset in ood_datasets.items():
				ood_iterator = iter(ood_dataset)
				logging.info('Calculating OOD on dataset %s', ood_dataset_name)
				logging.info('Running OOD eval at epoch: %s', epoch)
				test_step(ood_iterator, 'test', ood_dataset_name,
									steps_per_ood[ood_dataset_name])

				logging.info('Done with OOD eval on %s', ood_dataset_name)

		corrupt_results = {}
		if (FLAGS.corruptions_interval > 0 and
				(epoch + 1) % FLAGS.corruptions_interval == 0):
			corrupt_results = utils.aggregate_corrupt_metrics(corrupt_metrics, corruption_types)

		logging.info('Train Loss: %.4f, Accuracy: %.2f%%',
								 metrics['train/loss'].result(),
								 metrics['train/accuracy'].result() * 100)
		logging.info('Test NLL: %.4f, Accuracy: %.2f%%, ECE: %.4f',
								 metrics['test/negative_log_likelihood'].result(),
								 metrics['test/accuracy'].result() * 100,
								 metrics['test/ece'].result()['ece'],)
		total_results = {name: metric.result() for name, metric in metrics.items()}
		total_results.update(corrupt_results)
		# Metrics from Robustness Metrics (like ECE) will return a dict with a
		# single key/value, instead of a scalar.
		total_results = {
				k: (list(v.values())[0] if isinstance(v, dict) else v)
				for k, v in total_results.items()
		}
		with summary_writer.as_default():
			for name, result in total_results.items():
				tf.summary.scalar(name, result, step=epoch + 1)

		for metric in metrics.values():
			metric.reset_states()

		if FLAGS.corruptions_interval > 0:
			for metric in corrupt_metrics.values():
				metric.reset_states()

		if (FLAGS.checkpoint_interval > 0 and
				(epoch + 1) % FLAGS.checkpoint_interval == 0):
			checkpoint_name = checkpoint.save(
					os.path.join(FLAGS.output_dir, 'checkpoint'))
			logging.info('Saved checkpoint to %s', checkpoint_name)

	final_checkpoint_name = checkpoint.save(
      os.path.join(FLAGS.output_dir, 'checkpoint'))
	logging.info('Saved last checkpoint to %s', final_checkpoint_name)
	with summary_writer.as_default():
		hp.hparams({
				'base_learning_rate': FLAGS.base_learning_rate,
				'one_minus_momentum': FLAGS.one_minus_momentum,
				'l2': FLAGS.l2,
		})

	for epoch in range(0, 50):
		train_density(train_iterator)
		print("Density Loss: " + str(metrics['train/density_loss'].result()))
	density_model.save_weights(FLAGS.output_dir + '/density_model')
	# quit()

	density_model.load_weights(FLAGS.output_dir + '/density_model')
	train_nll = test_density(train_iterator, steps_per_epoch)
	train_nll = tf.reshape(train_nll,[-1])
	x_max = tf.reduce_max(train_nll)
	x_min = tf.reduce_min(train_nll)
	
	for epoch in range(0, 10):
		normalize_softmax(train_iterator)
		logging.info('Train Loss: %.4f, Accuracy: %.2f%%',
								 metrics['normalize/loss'].result(),
								 metrics['normalize/accuracy'].result() * 100)
	final_checkpoint_name = checkpoint.save(
      os.path.join(FLAGS.output_dir, 'checkpoint'))
	logging.info('Saved last checkpoint to %s', final_checkpoint_name)
	
	datasets_to_evaluate = {'clean': test_datasets['clean']}
	for dataset_name, test_dataset in datasets_to_evaluate.items():
		test_iterator = iter(test_dataset)
		test_ood_step(test_iterator, dataset_name, steps_per_eval, x_max, x_min)
		logging.info('Test NLL: %.4f, Accuracy: %.2f%%, ECE: %.4f',
								 metrics['test_ood/negative_log_likelihood'].result(),
								 metrics['test_ood/accuracy'].result() * 100,
								 metrics['test_ood/ece'].result()['ece'],)

if __name__ == '__main__':
	app.run(main)
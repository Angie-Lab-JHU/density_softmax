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

"""ResNet-50 on ImageNet trained with maximum likelihood and gradient descent.

This script supports using mixup [1], possibly combined with the rescaling of
the predictions proposed in [2] (see the metrics ending with `+rescaling`).
Mixup is enabled by setting ``mixup_alpha > 0`.

## References:

[1]: Hongyi Zhang et al. mixup: Beyond Empirical Risk Minimization.
		 _arXiv preprint arXiv:1710.09412_, 2017.
		 https://arxiv.org/abs/1710.09412
[2]: Luigi Carratino et al. On Mixup Regularization.
		 _arXiv preprint arXiv:2006.06049_, 2020.
		 https://arxiv.org/abs/2006.06049
"""

import os
import time

from absl import app
from absl import flags
from absl import logging
import numpy as np
import robustness_metrics as rm
import scipy
import tensorflow as tf
import tensorflow_datasets as tfds
import uncertainty_baselines as ub
from tensorboard.plugins.hparams import api as hp
import tensorflow_probability as tfp
import utils  # local file import from baselines.cifar

from tensorflow import keras
from tensorflow.keras import regularizers
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = "1"
# os.environ['TF_ENABLE_MKL_NATIVE_FORMAT'] = "1" 

flags.DEFINE_integer('per_core_batch_size', 128, 'Batch size per TPU core/GPU.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_float('base_learning_rate', 0.1,
									 'Base learning rate when train batch size is 256.')
flags.DEFINE_float('one_minus_momentum', 0.1, 'Optimizer momentum.')
flags.DEFINE_float('l2', 1e-4, 'L2 coefficient.')
flags.DEFINE_string('data_dir', None, 'Path to training and testing data.')
flags.DEFINE_string('output_dir', '/tmp/imagenet',
										'The directory where the model weights and '
										'training/evaluation summaries are stored.')
flags.DEFINE_integer('train_epochs', 90, 'Number of training epochs.')
flags.DEFINE_integer('checkpoint_interval', 25,
										 'Number of epochs between saving checkpoints. Use -1 to '
										 'never save checkpoints.')
flags.DEFINE_integer('num_bins', 15, 'Number of bins for ECE computation.')

# Mixup-related flags.
flags.DEFINE_float('mixup_alpha', 0., 'Coefficient of mixup distribution.')
flags.DEFINE_bool('same_mix_weight_per_batch', False,
									'Whether to use a single mix weight across the batch.')
flags.DEFINE_bool('use_random_shuffling', False,
									'Whether to use random shuffling to pair the points of mixup'
									'within a batch.')
flags.DEFINE_bool('use_truncated_beta', True,
									'Whether to sample the mixup weights from '
									'Beta[0,1](alpha,alpha) or from the truncated distribution '
									'Beta[1/2,1](alpha,alpha).')
flags.DEFINE_float(
		'train_proportion',
		1.0,
		'What proportion of the training set to use to train versus validate on.')

flags.DEFINE_string('corruption_type', None, 'corruption_type')
flags.DEFINE_integer('severity', None, 'Number')

# Accelerator flags.
flags.DEFINE_bool('use_gpu', False, 'Whether to run on GPU or otherwise TPU.')
flags.DEFINE_bool('use_bfloat16', False, 'Whether to use mixed precision.')
flags.DEFINE_integer('num_cores', 32, 'Number of TPU cores or number of GPUs.')
flags.DEFINE_string('tpu', None,
										'Name of the TPU. Only used if use_gpu is False.')
FLAGS = flags.FLAGS

# Number of images in ImageNet-1k train dataset.
APPROX_IMAGENET_TRAIN_IMAGES = 1281167
# Number of images in eval dataset.
IMAGENET_VALIDATION_IMAGES = 50000
NUM_CLASSES = 1000

_LR_SCHEDULE = [    # (multiplier, epoch to start) tuples
		(1.0, 5), (0.1, 30), (0.01, 60), (0.001, 80)
]

IMAGE_SHAPE = (224, 224, 3)


def mean_truncated_beta_distribution(alpha):
	"""Expectation of a truncated beta(alpha, alpha) distribution in [1/2, 1]."""
	return 1. - scipy.special.betainc(alpha + 1, alpha, .5)

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

def rescale(X, x_min, x_max, y_min = -0.25, y_max = -0.15):
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

	del argv  # unused arg
	tf.io.gfile.makedirs(FLAGS.output_dir)
	logging.info('Saving checkpoints at %s', FLAGS.output_dir)
	tf.random.set_seed(FLAGS.seed)

	batch_size = FLAGS.per_core_batch_size * FLAGS.num_cores

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

	enable_mixup = (FLAGS.mixup_alpha > 0.0)
	mixup_params = {
			'mixup_alpha': FLAGS.mixup_alpha,
			'adaptive_mixup': False,
			'same_mix_weight_per_batch': FLAGS.same_mix_weight_per_batch,
			'use_random_shuffling': FLAGS.use_random_shuffling,
			'use_truncated_beta': FLAGS.use_truncated_beta
	}

	train_builder = ub.datasets.ImageNetDataset(
			split=tfds.Split.TRAIN,
			use_bfloat16=FLAGS.use_bfloat16,
			one_hot=True,
			mixup_params=mixup_params,
			validation_percent=1.0 - FLAGS.train_proportion,
			data_dir=data_dir)
	train_dataset = train_builder.load(batch_size=batch_size, strategy=strategy)
	steps_per_epoch = train_builder.num_examples // batch_size

	test_builder = ub.datasets.ImageNetDataset(
			split=tfds.Split.TEST, use_bfloat16=FLAGS.use_bfloat16, data_dir=data_dir)
	test_dataset = test_builder.load(batch_size=batch_size, strategy=strategy)
	test_datasets = {
	#   'clean': test_dataset
  	}
	corruption_types, max_severity = utils.load_corrupted_test_info()
	dataset_name = '{0}_{1}'.format(FLAGS.corruption_type, FLAGS.severity)
	corrupted_builder = ub.datasets.ImageNetCorruptedDataset(
		corruption_type=FLAGS.corruption_type,
		severity=FLAGS.severity,
		use_bfloat16=FLAGS.use_bfloat16,
		download_data=True,
		data_dir=data_dir)
	test_datasets[dataset_name] = corrupted_builder.load(
		batch_size=batch_size, strategy=strategy)

	steps_per_test_eval = IMAGENET_VALIDATION_IMAGES // batch_size
	validation_dataset = None
	steps_per_validation_eval = 0
	if FLAGS.train_proportion < 1.0:
		# Note we do not one_hot the validation set.
		validation_builder = ub.datasets.ImageNetDataset(
				split=tfds.Split.VALIDATION,
				use_bfloat16=FLAGS.use_bfloat16,
				mixup_params=mixup_params,
				validation_percent=1.0 - FLAGS.train_proportion,
				data_dir=data_dir)
		validation_dataset = validation_builder.load(
				batch_size=batch_size, strategy=strategy)
		steps_per_validation_eval = validation_builder.num_examples // batch_size

	if FLAGS.use_bfloat16:
		tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')

	with strategy.scope():

		logging.info('Building Keras ResNet-50 model')
		model = ub.models.resnet50_deterministic(input_shape=IMAGE_SHAPE,
																						 num_classes=NUM_CLASSES)
		density_model = RealNVP(num_coupling_layers=2, input_dim = 2048)
		density_model.load_weights('checkpoints//density_softmax/imagenet/model2/density_model')
		encoder = tf.keras.Model(model.input, model.get_layer('avg_pool').output)
		classifer = tf.keras.Model(model.get_layer('fc1000').input, model.get_layer('fc1000').output)
		logging.info('Model input shape: %s', model.input_shape)
		logging.info('Model output shape: %s', model.output_shape)
		logging.info('Model number of weights: %s', model.count_params())
		# Scale learning rate and decay epochs by vanilla settings.
		base_lr = FLAGS.base_learning_rate * batch_size / 256
		decay_epochs = [
				(FLAGS.train_epochs * 30) // 90,
				(FLAGS.train_epochs * 60) // 90,
				(FLAGS.train_epochs * 80) // 90,
		]
		learning_rate = ub.schedules.WarmUpPiecewiseConstantSchedule(
				steps_per_epoch=steps_per_epoch,
				base_learning_rate=base_lr,
				decay_ratio=0.1,
				decay_epochs=decay_epochs,
				warmup_epochs=5)
		optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate,
																				momentum=1.0 - FLAGS.one_minus_momentum,
																				nesterov=True)
		metrics = {
				'train/negative_log_likelihood': tf.keras.metrics.Mean(),
				'train/accuracy': tf.keras.metrics.SparseCategoricalAccuracy(),
				'train/loss': tf.keras.metrics.Mean(),
				'train/density_loss': tf.keras.metrics.Mean(),
				'train/ece': rm.metrics.ExpectedCalibrationError(
						num_bins=FLAGS.num_bins),
				'test/negative_log_likelihood': tf.keras.metrics.Mean(),
				'test/accuracy': tf.keras.metrics.SparseCategoricalAccuracy(),
				'test/ece': rm.metrics.ExpectedCalibrationError(
						num_bins=FLAGS.num_bins),
		}
		if FLAGS.train_proportion < 1.0:
			metrics.update({
					'validation/negative_log_likelihood': tf.keras.metrics.Mean(),
					'validation/accuracy': tf.keras.metrics.SparseCategoricalAccuracy(),
					'validation/loss': tf.keras.metrics.Mean(),
					'validation/ece': rm.metrics.ExpectedCalibrationError(
							num_bins=FLAGS.num_bins),
			})
		logging.info('Finished building Keras ResNet-50 model')

		checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
		latest_checkpoint = tf.train.latest_checkpoint(FLAGS.output_dir)
		latest_checkpoint = latest_checkpoint.replace("density_model", "checkpoint-4")
		checkpoint.restore(latest_checkpoint)

	def compute_loss(density_model, x):
		y, logdet = density_model(x)
		log_likelihood = density_model.distribution.log_prob(y) + logdet
		return -tf.reduce_mean(log_likelihood)

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
	def update_test_metrics(
			labels, probs, metric_prefix='test', metric_suffix=''):
		negative_log_likelihood = tf.reduce_mean(
				tf.keras.losses.sparse_categorical_crossentropy(
						labels, probs))
		nll_key = metric_prefix + '/negative_log_likelihood' + metric_suffix
		metrics[nll_key].update_state(negative_log_likelihood)
		metrics[metric_prefix + '/accuracy' + metric_suffix].update_state(
				labels, probs)
		metrics[metric_prefix + '/ece' + metric_suffix].add_batch(
				probs, label=labels)

	@tf.function
	def test_step(metrics_prefix, iterator, steps_per_eval, x_max, x_min):
		"""Evaluation StepFn."""
		def step_fn(inputs):
			"""Per-Replica StepFn."""
			images = inputs['features']
			labels = inputs['labels']

			latents = encoder(images, training = False)
			log_likelihood = density_model.score_samples(latents)
			log_likelihood = rescale(log_likelihood, x_min, x_max)
			prob = tf.math.exp(log_likelihood)
			prob = tf.expand_dims(prob, 1)
			logits = classifer(latents)
			probs = tf.nn.softmax(logits * tf.cast(prob, dtype=tf.float32))

			if FLAGS.use_bfloat16:
				probs = tf.cast(probs, tf.float32)

			update_test_metrics(labels, probs, metric_prefix=metrics_prefix)

		for _ in tf.range(tf.cast(steps_per_eval, tf.int32)):
			strategy.run(step_fn, args=(next(iterator),))

	# train_iterator = iter(train_dataset)
	# train_nll = test_density(train_iterator, steps_per_epoch)
	# train_nll = tf.reshape(train_nll,[-1])
	# x_max = tf.reduce_max(train_nll)
	# x_min = tf.reduce_min(train_nll)

	# x_min = -134275530000000.0
	# x_max = -148.521
	
	x_min = -5534.878
	x_max = -221.95312
	# print(x_min)
	# print(x_max)
	cAcc, cECE, cNLL = [], [], []
	with open('out.npy', 'rb') as f:
		cNLL = np.load(f).tolist()
		cAcc = np.load(f).tolist()
		cECE = np.load(f).tolist()
	# print(cNLL)
	# print(cAcc)
	# print(cECE)
	# exit()
	
	for dataset_name, test_dataset in test_datasets.items():
		test_iterator = iter(test_dataset)
		test_step(
				metrics_prefix='test',
				iterator=test_iterator,
				steps_per_eval=steps_per_test_eval, x_max = x_max, x_min = x_min)
				
		logging.info('Test NLL: %.4f, Accuracy: %.2f%%, ECE: %.4f',
								 metrics['test/negative_log_likelihood'].result(),
								 metrics['test/accuracy'].result() * 100,
								 metrics['test/ece'].result()['ece'])

		if dataset_name != 'clean':
			cNLL.append(metrics['test/negative_log_likelihood'].result())
			cAcc.append(metrics['test/accuracy'].result() * 100)
			cECE.append(metrics['test/ece'].result()['ece'])

		for metric in metrics.values():
			metric.reset_states()

	cNLL = np.asarray(cNLL)
	cAcc = np.asarray(cAcc)
	cECE = np.asarray(cECE)
	print(cNLL)
	print(cAcc)
	print(cECE)

	with open('out.npy', 'wb') as f:
		np.save(f, cNLL)
		np.save(f, cAcc)
		np.save(f, cECE)

if __name__ == '__main__':
	app.run(main)

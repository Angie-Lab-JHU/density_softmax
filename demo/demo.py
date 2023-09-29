import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import pkg_resources
import importlib
importlib.reload(pkg_resources)

import matplotlib.pyplot as plt
import matplotlib.colors as colors

import sklearn.datasets

import numpy as np
import tensorflow as tf

import official.nlp.modeling.layers as nlp_layers
from sklearn.neighbors import KernelDensity
import edward2 as ed

from tensorflow import keras
import tensorflow_probability as tfp
from tensorflow.keras import layers
from tensorflow.keras import regularizers

# coding=utf-8
# Copyright 2022 The Uncertainty Baselines Authors.
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

"""Rank-1 BNN Utilities."""
import edward2 as ed
import numpy as np
import tensorflow as tf
import random

# seed_value = 6
# random.seed(seed_value)
# np.random.seed(seed_value)
# tf.random.set_seed(seed_value)

def _make_sign_initializer(random_sign_init):
	if random_sign_init > 0:
		return ed.initializers.RandomSign(random_sign_init)
	else:
		return tf.keras.initializers.RandomNormal(mean=1.0,
																							stddev=-random_sign_init)


def make_initializer(initializer, random_sign_init, dropout_rate):
	"""Builds initializer with specific mean and/or stddevs."""
	if initializer == 'trainable_deterministic':
		return ed.initializers.TrainableDeterministic(
				loc_initializer=_make_sign_initializer(random_sign_init))
	elif initializer == 'trainable_half_cauchy':
		stddev_init = np.log(np.expm1(np.sqrt(dropout_rate / (1. - dropout_rate))))
		return ed.initializers.TrainableHalfCauchy(
				loc_initializer=_make_sign_initializer(random_sign_init),
				scale_initializer=tf.keras.initializers.Constant(stddev_init),
				scale_constraint='softplus')
	elif initializer == 'trainable_cauchy':
		stddev_init = np.log(np.expm1(np.sqrt(dropout_rate / (1. - dropout_rate))))
		return ed.initializers.TrainableCauchy(
				loc_initializer=_make_sign_initializer(random_sign_init),
				scale_initializer=tf.keras.initializers.Constant(stddev_init),
				scale_constraint='softplus')
	elif initializer == 'trainable_normal':
		stddev_init = np.log(np.expm1(np.sqrt(dropout_rate / (1. - dropout_rate))))
		return ed.initializers.TrainableNormal(
				mean_initializer=_make_sign_initializer(random_sign_init),
				stddev_initializer=tf.keras.initializers.TruncatedNormal(
						mean=stddev_init, stddev=0.1),
				stddev_constraint='softplus')
	elif initializer == 'trainable_log_normal':
		stddev_init = np.log(np.expm1(np.sqrt(dropout_rate / (1. - dropout_rate))))
		return ed.initializers.TrainableLogNormal(
				loc_initializer=_make_sign_initializer(random_sign_init),
				scale_initializer=tf.keras.initializers.TruncatedNormal(
						mean=stddev_init, stddev=0.1),
				scale_constraint='softplus')
	elif initializer == 'trainable_normal_fixed_stddev':
		return ed.initializers.TrainableNormalFixedStddev(
				stddev=tf.sqrt(dropout_rate / (1. - dropout_rate)),
				mean_initializer=_make_sign_initializer(random_sign_init))
	elif initializer == 'trainable_normal_shared_stddev':
		stddev_init = np.log(np.expm1(np.sqrt(dropout_rate / (1. - dropout_rate))))
		return ed.initializers.TrainableNormalSharedStddev(
				mean_initializer=_make_sign_initializer(random_sign_init),
				stddev_initializer=tf.keras.initializers.Constant(stddev_init),
				stddev_constraint='softplus')
	return initializer


def make_regularizer(regularizer, mean, stddev):
	"""Builds regularizer with specific mean and/or stddevs."""
	if regularizer == 'normal_kl_divergence':
		return ed.regularizers.NormalKLDivergence(mean=mean, stddev=stddev)
	elif regularizer == 'log_normal_kl_divergence':
		return ed.regularizers.LogNormalKLDivergence(
				loc=tf.math.log(1.), scale=stddev)
	elif regularizer == 'normal_kl_divergence_with_tied_mean':
		return ed.regularizers.NormalKLDivergenceWithTiedMean(stddev=stddev)
	elif regularizer == 'cauchy_kl_divergence':
		return ed.regularizers.CauchyKLDivergence(loc=mean, scale=stddev)
	elif regularizer == 'normal_empirical_bayes_kl_divergence':
		return ed.regularizers.NormalEmpiricalBayesKLDivergence(mean=mean)
	elif regularizer == 'trainable_normal_kl_divergence_stddev':
		return ed.regularizers.TrainableNormalKLDivergenceStdDev(mean=mean)
	return regularizer
	
DEFAULT_X_RANGE = (-3, 3)
DEFAULT_Y_RANGE = (-2, 2)
DEFAULT_CMAP = colors.ListedColormap(["#377eb8", "#ff7f00"])
DEFAULT_NORM = colors.Normalize(vmin=0, vmax=1,)
DEFAULT_N_GRID = 100

def make_training_data(sample_size=500):
	"""Create two moon training dataset."""
	train_examples, train_labels = sklearn.datasets.make_moons(
			n_samples=2 * sample_size, noise=0.1)

	# Adjust data position slightly.
	train_examples[train_labels == 0] += [-0.1, 0.2]
	train_examples[train_labels == 1] += [0.1, -0.2]

	# train_examples_1 = np.random.multivariate_normal(
	#     (-1.5, 0), cov=np.diag((0.01, 0.005)), size=500)
	# train_examples_2 = np.random.multivariate_normal(
	#     (1.5, 0), cov=np.diag((0.01, 0.005)), size=500)
	# train_examples = np.concatenate((train_examples_1, train_examples_2), axis=0)
	# train_labels = np.concatenate((np.zeros(500), np.ones(500)))

	return train_examples, train_labels

def make_testing_data(x_range=DEFAULT_X_RANGE, y_range=DEFAULT_Y_RANGE, n_grid=DEFAULT_N_GRID):
	"""Create a mesh grid in 2D space."""
	# testing data (mesh grid over data space)
	x = np.linspace(x_range[0], x_range[1], n_grid)
	y = np.linspace(y_range[0], y_range[1], n_grid)
	xv, yv = np.meshgrid(x, y)
	return np.stack([xv.flatten(), yv.flatten()], axis=-1)

def make_ood_data(sample_size=500, means=(1, -1.5), vars=(0.1, 0.001)):
# def make_ood_data(sample_size=500, means=(0, -1.25), vars=(0.5, 0.001)):
	return np.random.multivariate_normal(
			means, cov=np.diag(vars), size=sample_size)

# Load the train, test and OOD datasets.
train_examples, train_labels = make_training_data(
		sample_size=500)
test_examples = make_testing_data()
ood_examples = make_ood_data(sample_size=500)

#Demo
#@title
class DeepResNet(tf.keras.Model):
	"""Defines a multi-layer residual network."""
	def __init__(self, num_classes, num_layers=3, num_hidden=128,
							 dropout_rate=0.1, **classifier_kwargs):
		super().__init__()
		# Defines class meta data.
		self.num_hidden = num_hidden
		self.num_layers = num_layers
		self.dropout_rate = dropout_rate
		self.classifier_kwargs = classifier_kwargs

		# Defines the hidden layers.
		self.input_layer = tf.keras.layers.Dense(self.num_hidden, trainable=False)
		self.dense_layers = [self.make_dense_layer() for _ in range(num_layers)]

		# Defines the output layer.
		self.classifier = self.make_output_layer(num_classes)

	def call(self, inputs):
		# Projects the 2d input data to high dimension.
		hidden = self.input_layer(inputs)

		# Computes the ResNet hidden representations.
		for i in range(self.num_layers):
			resid = self.dense_layers[i](hidden)
			resid = tf.keras.layers.Dropout(self.dropout_rate)(resid)
			hidden += resid

		return self.classifier(hidden)
	
	def encode(self, inputs):
		# Projects the 2d input data to high dimension.
		hidden = self.input_layer(inputs)

		# Computes the ResNet hidden representations.
		for i in range(self.num_layers):
			resid = self.dense_layers[i](hidden)
			resid = tf.keras.layers.Dropout(self.dropout_rate)(resid)
			hidden += resid
		
		return hidden

	def make_dense_layer(self):
		"""Uses the Dense layer as the hidden layer."""
		return tf.keras.layers.Dense(self.num_hidden, activation="relu")

	def make_output_layer(self, num_classes):
		"""Uses the Dense layer as the output layer."""
		return tf.keras.layers.Dense(
				num_classes, **self.classifier_kwargs)


#@title
def plot_uncertainty_surface(test_uncertainty, ax, cmap=None):
	"""Visualizes the 2D uncertainty surface.
	
	For simplicity, assume these objects already exist in the memory:

		test_examples: Array of test examples, shape (num_test, 2).
		train_labels: Array of train labels, shape (num_train, ).
		train_examples: Array of train examples, shape (num_train, 2).
	
	Arguments:
		test_uncertainty: Array of uncertainty scores, shape (num_test,).
		ax: A matplotlib Axes object that specifies a matplotlib figure.
		cmap: A matplotlib colormap object specifying the palette of the
			predictive surface.

	Returns:
		pcm: A matplotlib PathCollection object that contains the palette
			information of the uncertainty plot.
	"""
	# Normalize uncertainty for better visualization.
	test_uncertainty = test_uncertainty / np.max(test_uncertainty)

	# Set view limits.
	ax.set_ylim(DEFAULT_Y_RANGE)
	ax.set_xlim(DEFAULT_X_RANGE)

	# Plot normalized uncertainty surface.
	pcm = ax.imshow(
			np.reshape(test_uncertainty, [DEFAULT_N_GRID, DEFAULT_N_GRID]),
			cmap=cmap,
			origin="lower",
			extent=DEFAULT_X_RANGE + DEFAULT_Y_RANGE,
			vmin=DEFAULT_NORM.vmin,
			vmax=DEFAULT_NORM.vmax,
			interpolation='bicubic',
			aspect='auto')

	# Plot training data.
	ax.scatter(train_examples[:, 0], train_examples[:, 1],
						 c=train_labels, cmap=DEFAULT_CMAP, alpha=0.5)
	ax.scatter(ood_examples[:, 0], ood_examples[:, 1], c="red", alpha=0.1)

	return pcm

def create_deterministic():
	resnet_config = dict(num_classes=2, num_layers=6, num_hidden=128)
	resnet_model = DeepResNet(**resnet_config)

	resnet_model.build((None, 2))
	resnet_model.summary()

	loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
	metrics = tf.keras.metrics.SparseCategoricalAccuracy(),
	optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

	train_config = dict(loss=loss, metrics=metrics, optimizer=optimizer)

	fit_config = dict(batch_size=128, epochs=100)

	resnet_model.compile(**train_config)
	resnet_model.fit(train_examples, train_labels, **fit_config)

	resnet_logits = resnet_model(test_examples)
	resnet_probs = tf.nn.softmax(resnet_logits, axis=-1)[:, 0]
	resnet_uncertainty = resnet_probs * (1. - resnet_probs)

	# resnet_uncertainty = 1 - 2 * abs(resnet_probs - 0.5)

	# resnet_uncertainty = -resnet_probs * tf.math.log(resnet_probs) - (1. - resnet_probs) * tf.math.log((1. - resnet_probs))

	return resnet_probs, resnet_uncertainty

def create_dropout():
	def mc_dropout_sampling(test_examples):
		# Enable dropout during inference.
		return resnet_model(test_examples, training=True)

	resnet_config = dict(num_classes=2, num_layers=6, num_hidden=128)
	resnet_model = DeepResNet(**resnet_config)

	resnet_model.build((None, 2))
	resnet_model.summary()

	loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
	metrics = tf.keras.metrics.SparseCategoricalAccuracy(),
	optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

	train_config = dict(loss=loss, metrics=metrics, optimizer=optimizer)

	fit_config = dict(batch_size=128, epochs=100)

	resnet_model.compile(**train_config)
	resnet_model.fit(train_examples, train_labels, **fit_config)

	num_ensemble = 10
	dropout_logit_samples = [mc_dropout_sampling(test_examples) for _ in range(num_ensemble)]
	dropout_prob_samples = [tf.nn.softmax(dropout_logits, axis=-1)[:, 0] for dropout_logits in dropout_logit_samples]
	dropout_probs = tf.reduce_mean(dropout_prob_samples, axis=0)

	dropout_uncertainty = dropout_probs * (1. - dropout_probs)

	# dropout_uncertainty = 1 - 2 * abs(dropout_probs - 0.5)
	# dropout_uncertainty = -dropout_probs * tf.math.log(dropout_probs) - (1. - dropout_probs) * tf.math.log((1. - dropout_probs))

	return dropout_probs, dropout_uncertainty

def create_rank1():
	class DeepResNetRank1(tf.keras.Model):
		"""Defines a multi-layer residual network."""
		def __init__(self, num_classes, ensemble_size, num_layers=3, num_hidden=128,
								dropout_rate=0.1, **classifier_kwargs):
			super().__init__()
			# Defines class meta data.
			self.num_hidden = num_hidden
			self.num_layers = num_layers
			self.dropout_rate = dropout_rate
			self.classifier_kwargs = classifier_kwargs
			self.ensemble_size = ensemble_size
			self.alpha_initializer = "trainable_normal"
			self.random_sign_init = 0.5
			self.gamma_initializer = "trainable_normal"
			self.gamma_regularizer = "normal_kl_divergence"
			self.alpha_regularizer = "normal_kl_divergence"
			self.prior_mean = 1.
			self.prior_stddev = 0.1
			self.use_additive_perturbation = False

			# Defines the hidden layers.
			self.input_layer = ed.layers.DenseRank1(self.num_hidden,
				alpha_initializer=make_initializer(
						self.alpha_initializer, self.random_sign_init, self.dropout_rate),
				gamma_initializer=make_initializer(
						self.gamma_initializer, self.random_sign_init, self.dropout_rate),
				kernel_initializer='he_normal',
				alpha_regularizer=make_regularizer(
						self.alpha_regularizer, self.prior_mean, self.prior_stddev),
				gamma_regularizer=make_regularizer(
						self.gamma_regularizer, self.prior_mean, self.prior_stddev),
				use_additive_perturbation=self.use_additive_perturbation,
				ensemble_size = self.ensemble_size, trainable=False)
			# self.input_layer = tf.keras.layers.Dense(self.num_hidden, trainable=False)
			self.dense_layers = [self.make_dense_layer() for _ in range(num_layers)]

			# Defines the output layer.
			self.classifier = self.make_output_layer(num_classes)

		def call(self, inputs):
			# Projects the 2d input data to high dimension.
			hidden = self.input_layer(inputs)

			# Computes the ResNet hidden representations.
			for i in range(self.num_layers):
				resid = self.dense_layers[i](hidden)
				resid = tf.keras.layers.Dropout(self.dropout_rate)(resid)
				hidden += resid

			return self.classifier(hidden)
		
		def encode(self, inputs):
			# Projects the 2d input data to high dimension.
			hidden = self.input_layer(inputs)

			# Computes the ResNet hidden representations.
			for i in range(self.num_layers):
				resid = self.dense_layers[i](hidden)
				resid = tf.keras.layers.Dropout(self.dropout_rate)(resid)
				hidden += resid
			
			return hidden

		def make_dense_layer(self):
			"""Uses the Dense layer as the hidden layer."""
			# return tf.keras.layers.Dense(self.num_hidden, activation="relu")
			return ed.layers.DenseRank1(self.num_hidden, activation="relu",
				alpha_initializer=make_initializer(
						self.alpha_initializer, self.random_sign_init, self.dropout_rate),
				gamma_initializer=make_initializer(
						self.gamma_initializer, self.random_sign_init, self.dropout_rate),
				kernel_initializer='he_normal',
				alpha_regularizer=make_regularizer(
						self.alpha_regularizer, self.prior_mean, self.prior_stddev),
				gamma_regularizer=make_regularizer(
						self.gamma_regularizer, self.prior_mean, self.prior_stddev),
				use_additive_perturbation=self.use_additive_perturbation,
				ensemble_size = self.ensemble_size)

		def make_output_layer(self, num_classes):
			"""Uses the Dense layer as the output layer."""
			# return tf.keras.layers.Dense(
			#     num_classes, **self.classifier_kwargs)
			return ed.layers.DenseRank1(
					num_classes, **self.classifier_kwargs,
					alpha_initializer=make_initializer(
						self.alpha_initializer, self.random_sign_init, self.dropout_rate),
				gamma_initializer=make_initializer(
						self.gamma_initializer, self.random_sign_init, self.dropout_rate),
				kernel_initializer='he_normal',
				alpha_regularizer=make_regularizer(
						self.alpha_regularizer, self.prior_mean, self.prior_stddev),
				gamma_regularizer=make_regularizer(
						self.gamma_regularizer, self.prior_mean, self.prior_stddev),
				use_additive_perturbation=self.use_additive_perturbation,
				ensemble_size = self.ensemble_size)
	
	ensemble_size = 4
	resnet_config = dict(num_classes=2, num_layers=6, num_hidden=128, ensemble_size = ensemble_size)
	resnet_model = DeepResNetRank1(**resnet_config)

	resnet_model.build((None, 2))
	resnet_model.summary()

	loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
	metrics = tf.keras.metrics.SparseCategoricalAccuracy(),
	optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

	train_config = dict(loss=loss, metrics=metrics, optimizer=optimizer)

	fit_config = dict(batch_size=128, epochs=100)

	resnet_model.compile(**train_config)
	resnet_model.fit(tf.tile(train_examples, [ensemble_size, 1]), tf.tile(train_labels, [ensemble_size]), **fit_config)

	resnet_logits = []
	rank_1resnet_logits = resnet_model(tf.tile(test_examples, [ensemble_size, 1]))
	for i in range(len(test_examples)):
		tmp = []
		for j in range(ensemble_size):
				tmp.append(rank_1resnet_logits[i + (len(test_examples) * j)])
		resnet_logits.append(tf.reduce_mean(tmp, 0))
	
	resnet_probs_rank1 = tf.nn.softmax(resnet_logits, axis=-1)[:, 0]
	resnet_uncertainty_rank1 = resnet_probs_rank1 * (1. - resnet_probs_rank1)

	# resnet_uncertainty_rank1 = 1 - 2 * abs(resnet_probs_rank1 - 0.5)
	# resnet_uncertainty_rank1 = -resnet_probs_rank1 * tf.math.log(resnet_probs_rank1) - (1. - resnet_probs_rank1) * tf.math.log((1. - resnet_probs_rank1))

	return resnet_probs_rank1, resnet_uncertainty_rank1

def create_ensemble():
	resnet_config = dict(num_classes=2, num_layers=6, num_hidden=128)
	loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
	metrics = tf.keras.metrics.SparseCategoricalAccuracy(),
	optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
	fit_config = dict(batch_size=128, epochs=100)
	# Deep ensemble training
	num_ensemble = 10
	resnet_ensemble = []
	for _ in range(num_ensemble):
		resnet_model = DeepResNet(**resnet_config)
		resnet_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
		resnet_model.fit(train_examples, train_labels, verbose=0, **fit_config)
		resnet_ensemble.append(resnet_model)

	# Deep ensemble inference
	ensemble_logit_samples = [model(test_examples) for model in resnet_ensemble]
	ensemble_prob_samples = [tf.nn.softmax(logits, axis=-1)[:, 0] for logits in ensemble_logit_samples]
	ensemble_probs = tf.reduce_mean(ensemble_prob_samples, axis=0)
	ensemble_uncertainty = ensemble_probs * (1. - ensemble_probs)

	# ensemble_uncertainty = 1 - 2 * abs(ensemble_probs - 0.5)
	# ensemble_uncertainty = -ensemble_probs * tf.math.log(ensemble_probs) - (1. - ensemble_probs) * tf.math.log((1. - ensemble_probs))

	return ensemble_probs, ensemble_uncertainty

def create_sngp():
	resnet_config = dict(num_classes=2, num_layers=6, num_hidden=128)
	loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
	metrics = tf.keras.metrics.SparseCategoricalAccuracy(),
	optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

	train_config = dict(loss=loss, metrics=metrics, optimizer=optimizer)

	fit_config = dict(batch_size=128, epochs=100)

	dense = tf.keras.layers.Dense(units=10)
	dense = nlp_layers.SpectralNormalization(dense, norm_multiplier=0.9)

	batch_size = 32
	input_dim = 1024
	num_classes = 10

	gp_layer = nlp_layers.RandomFeatureGaussianProcess(units=num_classes,
																								num_inducing=1024,
																								normalize_input=False,
																								scale_random_features=True,
																								gp_cov_momentum=-1)

	embedding = tf.random.normal(shape=(batch_size, input_dim))
	logits, covmat = gp_layer(embedding)


	class DeepResNetSNGP(DeepResNet):
		def __init__(self, spec_norm_bound=0.9, **kwargs):
			self.spec_norm_bound = spec_norm_bound
			super().__init__(**kwargs)

		def make_dense_layer(self):
			"""Applies spectral normalization to the hidden layer."""
			dense_layer = super().make_dense_layer()
			return nlp_layers.SpectralNormalization(
					dense_layer, norm_multiplier=self.spec_norm_bound)

		def make_output_layer(self, num_classes):
			"""Uses Gaussian process as the output layer."""
			return nlp_layers.RandomFeatureGaussianProcess(
					num_classes,
					gp_cov_momentum=-1,
					**self.classifier_kwargs)

		def call(self, inputs, training=False, return_covmat=False):
			# Gets logits and a covariance matrix from the GP layer.
			logits, covmat = super().call(inputs)

			# Returns only logits during training.
			if not training and return_covmat:
				return logits, covmat

			return logits

	sngp_model = DeepResNetSNGP(**resnet_config)

	sngp_model.build((None, 2))
	sngp_model.summary()

	class ResetCovarianceCallback(tf.keras.callbacks.Callback):

		def on_epoch_begin(self, epoch, logs=None):
			"""Resets covariance matrix at the beginning of the epoch."""
			if epoch > 0:
				self.model.classifier.reset_covariance_matrix()

	class DeepResNetSNGPWithCovReset(DeepResNetSNGP):
		def fit(self, *args, **kwargs):
			"""Adds ResetCovarianceCallback to model callbacks."""
			kwargs["callbacks"] = list(kwargs.get("callbacks", []))
			kwargs["callbacks"].append(ResetCovarianceCallback())

			return super().fit(*args, **kwargs)

	sngp_model = DeepResNetSNGPWithCovReset(**resnet_config)
	sngp_model.compile(**train_config)
	sngp_model.fit(train_examples, train_labels, **fit_config)

	sngp_logits, sngp_covmat = sngp_model(test_examples, return_covmat=True)
	sngp_variance = tf.linalg.diag_part(sngp_covmat)[:, None]

	sngp_logits_adjusted = sngp_logits / tf.sqrt(1. + (np.pi / 8.) * sngp_variance)
	sngp_probs = tf.nn.softmax(sngp_logits_adjusted, axis=-1)[:, 0]


	def compute_posterior_mean_probability(logits, covmat, lambda_param=np.pi / 8.):
		# Computes uncertainty-adjusted logits using the built-in method.
		logits_adjusted = nlp_layers.gaussian_process.mean_field_logits(
				logits, covmat, mean_field_factor=lambda_param)
		
		return tf.nn.softmax(logits_adjusted, axis=-1)[:, 0]

	sngp_logits, sngp_covmat = sngp_model(test_examples, return_covmat=True)
	sngp_probs = compute_posterior_mean_probability(sngp_logits, sngp_covmat)

	def train_and_test_sngp(train_examples, test_examples):
		sngp_model = DeepResNetSNGPWithCovReset(**resnet_config)

		sngp_model.compile(**train_config)
		sngp_model.fit(train_examples, train_labels, verbose=0, **fit_config)

		sngp_logits, sngp_covmat = sngp_model(test_examples, return_covmat=True)
		sngp_probs = compute_posterior_mean_probability(sngp_logits, sngp_covmat)

		return sngp_probs

	sngp_probs = train_and_test_sngp(train_examples, test_examples)
	sngp_uncertainty = sngp_probs * (1. - sngp_probs)

	# sngp_uncertainty = 1 - 2 * abs(sngp_probs - 0.5)
	# sngp_uncertainty = -sngp_probs * (tf.math.log(sngp_probs)/tf.math.log(2.)) - (1. - sngp_probs) * (tf.math.log((1. - sngp_probs))/tf.math.log(2.))

	return sngp_probs, sngp_uncertainty

def create_density_softmax():
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

				self.distribution = tfp.distributions.MultivariateNormalDiag(
					loc=np.zeros(input_dim, dtype="float32"), scale_diag=np.ones(input_dim, dtype="float32")
				)
				self.masks = np.array(
					[np.concatenate((np.zeros(input_dim // 2), np.ones(input_dim // 2))), np.concatenate((np.ones(input_dim // 2), np.zeros(input_dim // 2)))] * (num_coupling_layers // 2), dtype="float32"
				)

				self.loss_tracker = keras.metrics.Mean(name="loss")
				self.layers_list = [Coupling(input_dim) for i in range(num_coupling_layers)]

			@property
			def metrics(self):
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

			def train_step(self, data):
					with tf.GradientTape() as tape:

							loss = self.log_loss(data)

					g = tape.gradient(loss, self.trainable_variables)
					self.optimizer.apply_gradients(zip(g, self.trainable_variables))
					self.loss_tracker.update_state(loss)

					return {"loss": self.loss_tracker.result()}

			def test_step(self, data):
					loss = self.log_loss(data)
					self.loss_tracker.update_state(loss)

					return {"loss": self.loss_tracker.result()}

	def rescale(X, x_min, x_max, y_min, y_max):
		X = (X - x_min) / (x_max - x_min)
		X = X * (y_max - y_min)
		X = X + y_min
		return tf.cast(X, tf.float64)

	resnet_config = dict(num_classes=2, num_layers=6, num_hidden=128)
	resnet_model = DeepResNet(**resnet_config)

	resnet_model.build((None, 2))
	resnet_model.summary()

	batch_size, epochs = 128, 100
	grad_penalty = 0.01

	loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
	train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
	optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

	train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
	train_dataset = train_dataset.shuffle(batch_size).batch(batch_size)

	@tf.function
	def train_step(x, y):
		with tf.GradientTape(persistent = True) as tape:
			tape.watch(x)
			hidden = resnet_model.encode(x)
			logits = resnet_model.classifier(hidden)
			loss_value = tf.cast(loss_fn(y, logits), tf.float64)

			grad_norm = tf.sqrt(tf.reduce_sum(tape.batch_jacobian(hidden, x) ** 2, axis = [1, 2]))
			loss_value += (grad_penalty * tf.reduce_mean(((grad_norm - 1) ** 2)))

		grads = tape.gradient(loss_value, resnet_model.trainable_weights)
		optimizer.apply_gradients(zip(grads, resnet_model.trainable_weights))
		train_acc_metric.update_state(y, logits)
		return loss_value

	start_time = time.time()
	for e in range(epochs):
		for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
			loss_value = train_step(x_batch_train, y_batch_train)
		
		print("Training loss at epoch %d: %.4f"% (e, float(loss_value)))
		# Display metrics at the end of each epoch.
		train_acc = train_acc_metric.result()
		print("Training acc over epoch: %.4f" % (float(train_acc),))

		# Reset training metrics at the end of each epoch
		train_acc_metric.reset_states()

	flow_model = RealNVP(num_coupling_layers=2, input_dim = 128)
	flow_model.compile(optimizer=keras.optimizers.Adam())

	train_latents = resnet_model.encode(train_examples)
	flow_model.fit(train_latents, batch_size=len(train_examples), epochs=300, verbose=0)

	train_logll = flow_model.score_samples(train_latents)
	x_min = tf.reduce_min(train_logll)
	x_max = tf.reduce_max(train_logll)

	train_likelihood = tf.exp(rescale(train_logll, x_min, x_max, -1, 1))
	train_likelihood = tf.expand_dims(train_likelihood, 1)
	train_likelihood_max = tf.reduce_max(train_likelihood)

	for e in range(1):
		for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
			loss_value = 0
			with tf.GradientTape() as tape:
				hidden = resnet_model.encode(x_batch_train)
				likelihood = tf.exp(rescale(flow_model.score_samples(hidden), x_min, x_max, -1, 1))
				likelihood = tf.expand_dims(likelihood, 1)
				likelihood = (likelihood)/(train_likelihood_max)
				logits = resnet_model.classifier(hidden) * tf.cast(likelihood, dtype=tf.float32)
				loss_value = loss_fn(y_batch_train, logits)

			grads = tape.gradient(loss_value, resnet_model.classifier.trainable_weights)
			optimizer.apply_gradients(zip(grads, resnet_model.classifier.trainable_weights))
			train_acc_metric.update_state(y_batch_train, logits)
		
		print("Training loss at epoch %d: %.4f"% (e, float(loss_value)))
		# Display metrics at the end of each epoch.
		train_acc = train_acc_metric.result()
		print("Training acc over epoch: %.4f" % (float(train_acc),))

		# Reset training metrics at the end of each epoch
		train_acc_metric.reset_states()

	density_softmax_training_time = (time.time() - start_time) * 1e6 / 128

	tstart_time = time.time()
	test_latents = resnet_model.encode(test_examples)
	resnet_logits = resnet_model.classifier(test_latents)
	test_likelihood = tf.exp(rescale(flow_model.score_samples(test_latents), x_min, x_max, -1, 1))
	test_likelihood = tf.expand_dims(test_likelihood, 1)

	test_likelihood = (test_likelihood)/(train_likelihood_max)
	dst_probs = tf.nn.softmax(resnet_logits * tf.cast(test_likelihood, dtype=tf.float32), axis=-1)[:, 0]
	density_softmax_infer_time = (time.time() - start_time) * 1e6
	dst_uncertainty = dst_probs * (1. - dst_probs)

	# dst_uncertainty = 1 - 2 * abs(dst_probs - 0.5)

	# dst_uncertainty = -dst_probs * tf.math.log(dst_probs) - (1. - dst_probs) * tf.math.log((1. - dst_probs))
	
	return dst_probs, dst_uncertainty

resnet_probs, resnet_uncertainty = create_deterministic()
dropout_probs, dropout_uncertainty = create_dropout()
rank1_probs, rank1_uncertainty = create_rank1()
ensemble_probs, ensemble_uncertainty = create_ensemble()
sngp_probs, sngp_uncertainty = create_sngp()
dst_probs, dst_uncertainty = create_density_softmax()

fig, axs = plt.subplots(2, 6, figsize = (24, 8), constrained_layout=True)
pcm_0 = plot_uncertainty_surface(resnet_probs, ax=axs[0][0])
pcm_1 = plot_uncertainty_surface(resnet_uncertainty, ax=axs[1][0])
axs[0][0].set_title("ERM",fontsize=18)
axs[1][0].set_title("ERM",fontsize=18) 

axs[0][0].set_ylabel('Class Probability',fontsize=18)
axs[1][0].set_ylabel('Predictive Uncertainty',fontsize=18)

pcm_0 = plot_uncertainty_surface(dropout_probs, ax=axs[0][1])
pcm_1 = plot_uncertainty_surface(dropout_uncertainty, ax=axs[1][1])
axs[0][1].set_title("MC Dropout",fontsize=18)
axs[1][1].set_title("MC Dropout",fontsize=18) 

pcm_0 = plot_uncertainty_surface(rank1_probs, ax=axs[0][2])
pcm_1 = plot_uncertainty_surface(rank1_uncertainty, ax=axs[1][2])
axs[0][2].set_title("Rank-1 BNN",fontsize=18)
axs[1][2].set_title("Rank-1 BNN",fontsize=18) 

pcm_0 = plot_uncertainty_surface(ensemble_probs, ax=axs[0][3])
pcm_1 = plot_uncertainty_surface(ensemble_uncertainty, ax=axs[1][3])
axs[0][3].set_title("Ensemble",fontsize=18)
axs[1][3].set_title("Ensemble",fontsize=18) 

pcm_0 = plot_uncertainty_surface(sngp_probs, ax=axs[0][4])
pcm_1 = plot_uncertainty_surface(sngp_uncertainty, ax=axs[1][4])
axs[0][4].set_title("SNGP",fontsize=18)
axs[1][4].set_title("SNGP",fontsize=18) 

pcm_0 = plot_uncertainty_surface(dst_probs, ax=axs[0][5])
pcm_1 = plot_uncertainty_surface(dst_uncertainty, ax=axs[1][5])
axs[0][5].set_title("Density Softmax",fontsize=18)
axs[1][5].set_title("Density Softmax",fontsize=18)

fig.colorbar(pcm_0, ax=axs[0:, :], shrink=0.6, pad = 0.015)

# plt.tight_layout()
plt.savefig("out.pdf")
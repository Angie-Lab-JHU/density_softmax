import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
from sklearn.mixture import GaussianMixture
import edward2 as ed
import random
from tensorflow import keras
import time


import tensorflow_probability as tfp
from tensorflow.keras import layers
from tensorflow.keras import regularizers

# 4
# seed_value = 25
# random.seed(seed_value)
# np.random.seed(seed_value)
# tf.random.set_seed(seed_value)

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
	#     (-1, 0), cov=np.diag((0.1, 0.001)), size=500)
	# train_examples_2 = np.random.multivariate_normal(
	#     (1, 0), cov=np.diag((0.1, 0.001)), size=500)
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
		hidden = self.encode(inputs)

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

def create_density_softmax():
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

	start_time = time.time()
	test_latents = resnet_model.encode(test_examples)
	resnet_logits = resnet_model.classifier(test_latents)
	test_likelihood = tf.exp(rescale(flow_model.score_samples(test_latents), x_min, x_max, -1, 1))
	test_likelihood = tf.expand_dims(test_likelihood, 1)

	test_likelihood = (test_likelihood)/(train_likelihood_max)
	dst_probs = tf.nn.softmax(resnet_logits * tf.cast(test_likelihood, dtype=tf.float32), axis=-1)[:, 0]
	density_softmax_infer_time = (time.time() - start_time) * 1e6

	dst_uncertainty = dst_probs * (1. - dst_probs)
	return dst_probs, dst_uncertainty

dst_probs, dst_uncertainty = create_density_softmax()

fig, axs = plt.subplots(1, 3, figsize = (12, 4), constrained_layout=True)

pos_examples = train_examples[train_labels == 0]
neg_examples = train_examples[train_labels == 1]

axs[0].scatter(pos_examples[:, 0], pos_examples[:, 1], c="#377eb8", alpha=0.5)
axs[0].scatter(neg_examples[:, 0], neg_examples[:, 1], c="#ff7f00", alpha=0.5)
axs[0].scatter(ood_examples[:, 0], ood_examples[:, 1], c="red", alpha=0.1)

axs[0].legend(["Positive", "Negative", "Out-of-Domain"])

axs[0].set_ylim(DEFAULT_Y_RANGE)
axs[0].set_xlim(DEFAULT_X_RANGE)
axs[0].set_title("Data Visualization", fontsize=18)


pcm_0 = plot_uncertainty_surface(dst_probs, ax=axs[1])
pcm_1 = plot_uncertainty_surface(dst_uncertainty, ax=axs[2])
axs[1].set_title("Class Probability", fontsize=18)
axs[2].set_title("Predictive Uncertainty", fontsize=18)

fig.colorbar(pcm_0, ax=axs[2])

# plt.tight_layout()
plt.savefig("out.png")
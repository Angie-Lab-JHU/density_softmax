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
import edward2 as ed
from keras.utils.layer_utils import count_params
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

seed_value = 6
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

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

#   train_examples_1 = np.random.multivariate_normal(
#       (-1.5, 0), cov=np.diag((0.01, 0.005)), size=500)
#   train_examples_2 = np.random.multivariate_normal(
#       (1.5, 0), cov=np.diag((0.01, 0.005)), size=500)
#   train_examples = np.concatenate((train_examples_1, train_examples_2), axis=0)
#   train_labels = np.concatenate((np.zeros(500), np.ones(500)))

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

  return resnet_probs, resnet_uncertainty

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

  return sngp_probs, sngp_uncertainty

def create_gp():
  resnet_config = dict(num_classes=2, num_layers=6, num_hidden=128)
  loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  metrics = tf.keras.metrics.SparseCategoricalAccuracy(),
  optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

  train_config = dict(loss=loss, metrics=metrics, optimizer=optimizer)

  fit_config = dict(batch_size=128, epochs=100)

  class DeepResNetGP(DeepResNet):
    def __init__(self, **kwargs):
      super().__init__(**kwargs)

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

  class ResetCovarianceCallback(tf.keras.callbacks.Callback):

    def on_epoch_begin(self, epoch, logs=None):
      """Resets covariance matrix at the beginning of the epoch."""
      if epoch > 0:
        self.model.classifier.reset_covariance_matrix()

  class DeepResNetGPWithCovReset(DeepResNetGP):
    def fit(self, *args, **kwargs):
      """Adds ResetCovarianceCallback to model callbacks."""
      kwargs["callbacks"] = list(kwargs.get("callbacks", []))
      kwargs["callbacks"].append(ResetCovarianceCallback())

      return super().fit(*args, **kwargs)

  def compute_posterior_mean_probability(logits, covmat, lambda_param=np.pi / 8.):
    # Computes uncertainty-adjusted logits using the built-in method.
    logits_adjusted = nlp_layers.gaussian_process.mean_field_logits(
        logits, covmat, mean_field_factor=lambda_param)
    
    return tf.nn.softmax(logits_adjusted, axis=-1)[:, 0]

  def train_and_test_sngp(train_examples, test_examples):
    gp_model = DeepResNetGPWithCovReset(**resnet_config)
    gp_model.build((None, 2))
    gp_model.summary()

    gp_model.compile(**train_config)
    gp_model.fit(train_examples, train_labels, verbose=0, **fit_config)

    gp_logits, gp_covmat = gp_model(test_examples, return_covmat=True)
    gp_probs = compute_posterior_mean_probability(gp_logits, gp_covmat)

    return gp_probs

  gp_probs = train_and_test_sngp(train_examples, test_examples)
  gp_uncertainty = gp_probs * (1. - gp_probs)

  return gp_probs, gp_uncertainty

def create_sn():
  resnet_config = dict(num_classes=2, num_layers=6, num_hidden=128)
  loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  metrics = tf.keras.metrics.SparseCategoricalAccuracy(),
  optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

  train_config = dict(loss=loss, metrics=metrics, optimizer=optimizer)

  fit_config = dict(batch_size=128, epochs=100)

  class DeepResNetSN(DeepResNet):
    def __init__(self, spec_norm_bound=0.9, **kwargs):
      self.spec_norm_bound = spec_norm_bound
      super().__init__(**kwargs)

    def make_dense_layer(self):
      """Applies spectral normalization to the hidden layer."""
      dense_layer = super().make_dense_layer()
      return nlp_layers.SpectralNormalization(
          dense_layer, norm_multiplier=self.spec_norm_bound)

  sn_model = DeepResNetSN(**resnet_config)
  sn_model.build((None, 2))
  sn_model.summary()
  sn_model.compile(**train_config)
  sn_model.fit(train_examples, train_labels, **fit_config)

  sn_logits = sn_model(test_examples)
  sn_probs = tf.nn.softmax(sn_logits, axis=-1)[:, 0]
  sn_uncertainty = sn_probs * (1. - sn_probs)

  return sn_probs, sn_uncertainty

def create_sn_density_softmax():
  resnet_config = dict(num_classes=2, num_layers=6, num_hidden=128)
  class DeepResNetSN(DeepResNet):
    def __init__(self, spec_norm_bound=0.9, **kwargs):
      self.spec_norm_bound = spec_norm_bound
      super().__init__(**kwargs)

    def make_dense_layer(self):
      """Applies spectral normalization to the hidden layer."""
      dense_layer = super().make_dense_layer()
      return nlp_layers.SpectralNormalization(
          dense_layer, norm_multiplier=self.spec_norm_bound)

  resnet_model = DeepResNetSN(**resnet_config)

  resnet_model.build((None, 2))
  resnet_model.summary()

  loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  metrics = tf.keras.metrics.SparseCategoricalAccuracy(),
  optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

  train_config = dict(loss=loss, metrics=metrics, optimizer=optimizer)

  fit_config = dict(batch_size=128, epochs=100)

  resnet_model.compile(**train_config)
  resnet_model.fit(train_examples, train_labels, **fit_config)
  resnet_model.build((None, 2))
  resnet_model.summary()

  train_latents = resnet_model.encode(train_examples)
  kde = KernelDensity(kernel='gaussian', bandwidth = 1.5).fit(train_latents)

  # likelihood = tf.exp(model.(train_examples))
  likelihood = tf.exp(kde.score_samples(train_latents))
  likelihood = tf.expand_dims(likelihood, 1)
  ll_max = tf.reduce_max(likelihood)
  ll_min = tf.reduce_min(likelihood)
  print(ll_max)
  print(ll_min)
  # quit()

  test_latents = resnet_model.encode(test_examples)
  resnet_logits = resnet_model.classifier(test_latents)
  likelihood = tf.exp(kde.score_samples(test_latents))
  likelihood = tf.expand_dims(likelihood, 1)

  likelihood = (likelihood)/(ll_max)
  print(tf.reduce_max(likelihood))
  print(tf.reduce_min(likelihood))
  dst_probs = tf.nn.softmax(resnet_logits * tf.cast(likelihood, dtype=tf.float32), axis=-1)[:, 0]
  dst_uncertainty = dst_probs * (1. - dst_probs)

  return dst_probs, dst_uncertainty

def create_density_softmax():
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

  train_latents = resnet_model.encode(train_examples)
  kde = KernelDensity(kernel='gaussian', bandwidth = 2.0).fit(train_latents)

  # likelihood = tf.exp(model.(train_examples))
  likelihood = tf.exp(kde.score_samples(train_latents))
  likelihood = tf.expand_dims(likelihood, 1)
  ll_max = tf.reduce_max(likelihood)
  ll_min = tf.reduce_min(likelihood)
  print(ll_max)
  print(ll_min)
  # quit()

  test_latents = resnet_model.encode(test_examples)
  resnet_logits = resnet_model.classifier(test_latents)
  likelihood = tf.exp(kde.score_samples(test_latents))
  likelihood = tf.expand_dims(likelihood, 1)

  likelihood = (likelihood)/(ll_max)
  print(tf.reduce_max(likelihood))
  print(tf.reduce_min(likelihood))
  dst_probs = tf.nn.softmax(resnet_logits * tf.cast(likelihood, dtype=tf.float32), axis=-1)[:, 0]
  dst_uncertainty = dst_probs * (1. - dst_probs)

  return dst_probs, dst_uncertainty

resnet_probs, resnet_uncertainty = create_deterministic()
sngp_probs, sngp_uncertainty = create_sngp()
sn_probs, sn_uncertainty = create_sn()
gp_probs, gp_uncertainty = create_gp()
sndst_probs, sndst_uncertainty = create_sn_density_softmax()
dst_probs, dst_uncertainty = create_density_softmax()

fig, axs = plt.subplots(2, 6, figsize = (24, 8), constrained_layout=True)
pcm_0 = plot_uncertainty_surface(resnet_probs, ax=axs[0][0])
pcm_1 = plot_uncertainty_surface(resnet_uncertainty, ax=axs[1][0])
axs[0][0].set_title("ERM",fontsize=18)
axs[1][0].set_title("ERM",fontsize=18) 

axs[0][0].set_ylabel('Class Probability',fontsize=18)
axs[1][0].set_ylabel('Predictive Uncertainty',fontsize=18)

pcm_0 = plot_uncertainty_surface(sngp_probs, ax=axs[0][1])
pcm_1 = plot_uncertainty_surface(sngp_uncertainty, ax=axs[1][1])
axs[0][1].set_title("SNGP",fontsize=18)
axs[1][1].set_title("SNGP",fontsize=18) 

pcm_0 = plot_uncertainty_surface(sn_probs, ax=axs[0][2])
pcm_1 = plot_uncertainty_surface(sn_uncertainty, ax=axs[1][2])
axs[0][2].set_title("DNN-SN",fontsize=18)
axs[1][2].set_title("DNN-SN",fontsize=18) 

pcm_0 = plot_uncertainty_surface(gp_probs, ax=axs[0][3])
pcm_1 = plot_uncertainty_surface(gp_uncertainty, ax=axs[1][3])
axs[0][3].set_title("DNN-GP",fontsize=18)
axs[1][3].set_title("DNN-GP",fontsize=18) 

pcm_0 = plot_uncertainty_surface(sndst_probs, ax=axs[0][4])
pcm_1 = plot_uncertainty_surface(sndst_uncertainty, ax=axs[1][4])
axs[0][4].set_title("SN-Density Softmax",fontsize=18)
axs[1][4].set_title("SN-Density Softmax",fontsize=18) 

pcm_0 = plot_uncertainty_surface(dst_probs, ax=axs[0][5])
pcm_1 = plot_uncertainty_surface(dst_uncertainty, ax=axs[1][5])
axs[0][5].set_title("Density Softmax",fontsize=18)
axs[1][5].set_title("Density Softmax",fontsize=18) 

fig.colorbar(pcm_0, ax=axs[0:, :], shrink=0.6, pad = 0.015)

# plt.tight_layout()
plt.savefig("out/distance_preserving.pdf")
import logging
import math
import os
import pickle
import shutil

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from algorithms.Flows.src.dataloaders import dataloader_factory
from algorithms.Flows.src.models import model_factory
from scipy.stats import entropy
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.load_metadata import set_tr_val_samples_labels, set_test_samples_labels
from torchmetrics.functional.classification import multiclass_calibration_error
from torch.optim.lr_scheduler import MultiStepLR
import xlwt
from xlwt import Workbook
import utils.flows as fnn


class Classifier(nn.Module):
	def __init__(self, feature_dim, classes):
		super(Classifier, self).__init__()
		self.classifier = nn.Linear(feature_dim, classes)

	def forward(self, z):
		y = self.classifier(z)
		return y

class Trainer_Flows:
	def __init__(self, args, device, exp_idx):
		self.args = args
		self.device = device
		self.exp_idx = exp_idx
		self.checkpoint_name = (
			"algorithms/" + self.args.algorithm + "/results/checkpoints/" + self.args.exp_name + "_" + self.exp_idx
		)
		self.plot_dir = (
			"algorithms/" + self.args.algorithm + "/results/plots/" + self.args.exp_name + "_" + self.exp_idx + "/"
		)
		self.model = model_factory.get_model(self.args.model)().to(self.device)
		self.classifier = Classifier(self.args.feature_dim, self.args.n_classes).to(self.device)
		
		modules = []
		mask = torch.arange(0, self.args.feature_dim) % 2
		mask = mask.to(self.device).float()
		for _ in range(5):
			modules += [
				fnn.MADE(self.args.feature_dim, self.args.feature_dim, None, act="relu"),
				fnn.BatchNormFlow(self.args.feature_dim),
				fnn.Reverse(self.args.feature_dim)
			]
		self.flows = fnn.FlowSequential(*modules).to(self.device)
		for module in self.flows.modules():
			if isinstance(module, nn.Linear):
				nn.init.orthogonal_(module.weight)
				if hasattr(module, 'bias') and module.bias is not None:
					module.bias.data.fill_(0)

		self.nn_logsoftmax = nn.LogSoftmax(dim=1)
		self.criterion = nn.NLLLoss()

	def set_writer(self, log_dir):
		if not os.path.exists(log_dir):
			os.mkdir(log_dir)
		shutil.rmtree(log_dir)
		return SummaryWriter(log_dir)

	def init_training(self):
		logging.basicConfig(
			filename="algorithms/"
			+ self.args.algorithm
			+ "/results/logs/"
			+ self.args.exp_name
			+ "_"
			+ self.exp_idx
			+ ".log",
			filemode="w",
			level=logging.INFO,
		)
		self.writer = self.set_writer(
			log_dir="algorithms/"
			+ self.args.algorithm
			+ "/results/tensorboards/"
			+ self.args.exp_name
			+ "_"
			+ self.exp_idx
			+ "/"
		)
		(
			tr_sample_paths,
			tr_class_labels,
			val_sample_paths,
			val_class_labels,
		) = set_tr_val_samples_labels(self.args.train_meta_filenames, self.args.val_size)
		self.train_loader = DataLoader(
			dataloader_factory.get_train_dataloader(self.args.dataset)(
				path=self.args.train_path,
				sample_paths=tr_sample_paths,
				class_labels=tr_class_labels,
			),
			batch_size=self.args.batch_size,
			shuffle=True,
		)
		if self.args.val_size != 0:
			self.val_loader = DataLoader(
				dataloader_factory.get_test_dataloader(self.args.dataset)(
					path=self.args.train_path,
					sample_paths=val_sample_paths,
					class_labels=val_class_labels,
				),
				batch_size=self.args.batch_size,
				shuffle=True,
			)
		else:
			self.val_loader = DataLoader(
				dataloader_factory.get_test_dataloader(self.args.dataset)(
					path=self.args.train_path, sample_paths=test_sample_paths, class_labels=test_class_labels
				),
				batch_size=self.args.batch_size,
				shuffle=True,
			)
		optimizer_params = list(self.model.parameters()) + list(self.classifier.parameters())
		self.optimizer = torch.optim.SGD(optimizer_params, lr=self.args.learning_rate, momentum=0.9, nesterov=True)
		self.scheduler = MultiStepLR(self.optimizer, milestones=self.args.decay_interations, gamma=0.2)
		self.density_optimizer = torch.optim.Adam(self.flows.parameters(), lr=0.0001, weight_decay=1e-6)
		self.val_loss_min = np.Inf
		self.val_acc_max = 0
		self.calibrate_optimizer = torch.optim.SGD(self.classifier.parameters(), lr=self.args.learning_rate, momentum=0.9, nesterov=True)
		self.calibrate_scheduler = MultiStepLR(self.calibrate_optimizer, milestones=self.args.decay_interations, gamma=0.2)

	def train(self):
		self.init_training()
		self.model.train()
		self.classifier.train()
		n_class_corrected, total_classification_loss, total_samples = 0, 0, 0
		self.train_iter_loader = iter(self.train_loader)
		for iteration in range(self.args.iterations):
			if (iteration % len(self.train_iter_loader)) == 0:
				self.train_iter_loader = iter(self.train_loader)
			samples, labels = self.train_iter_loader.next()
			samples, labels = samples.to(self.device), labels.to(self.device)
			predicted_softmaxs = self.nn_logsoftmax(self.classifier(self.model(samples)))
			classification_loss = self.criterion(predicted_softmaxs, labels)
			total_classification_loss += classification_loss.item()
			_, predicted_softmaxs = torch.max(predicted_softmaxs, 1)
			n_class_corrected += (predicted_softmaxs == labels).sum().item()
			total_samples += len(samples)
			self.optimizer.zero_grad()
			classification_loss.backward()
			self.optimizer.step()
			self.scheduler.step()
			if iteration % self.args.step_eval == (self.args.step_eval - 1):
				self.writer.add_scalar("Accuracy/train", 100.0 * n_class_corrected / total_samples, iteration)
				self.writer.add_scalar("Loss/train", total_classification_loss / self.args.step_eval, iteration)
				logging.info(
					"Train set: Iteration: [{}/{}]\tAccuracy: {}/{} ({:.2f}%)\tLoss: {:.6f}".format(
						iteration + 1,
						self.args.iterations,
						n_class_corrected,
						total_samples,
						100.0 * n_class_corrected / total_samples,
						total_classification_loss / self.args.step_eval,
					)
				)
				self.evaluate(iteration)
				n_class_corrected, total_classification_loss, total_samples = 0, 0, 0

		self.estimate_density()
		do_calib = True
		if do_calib == True:
			self.val_loss_min = np.Inf
			self.val_acc_max = 0
			checkpoint = torch.load(self.checkpoint_name + ".pt")
			self.model.load_state_dict(checkpoint["model_state_dict"])
			self.classifier.load_state_dict(checkpoint["classifier_state_dict"])
			checkpoint_score = torch.load(self.checkpoint_name + "_score.pt")
			self.flows.load_state_dict(checkpoint_score["score_state_dict"])
			self.model.eval()
			self.flows.eval()
			self.classifier.train()
			n_class_corrected, total_classification_loss, total_samples = 0, 0, 0
			for iteration in range(self.args.iterations):
				if (iteration % len(self.train_iter_loader)) == 0:
					self.train_iter_loader = iter(self.train_loader)
				samples, labels = self.train_iter_loader.next()
				samples, labels = samples.to(self.device), labels.to(self.device)
				latents = self.model(samples)
				prob = torch.sigmoid(torch.exp(self.flows.log_probs(latents)))
				predicted_classes = self.classifier(latents)
				predicted_softmaxs = self.nn_logsoftmax(prob * predicted_classes)

				classification_loss = self.criterion(predicted_softmaxs, labels)
				total_classification_loss += classification_loss.item()
				_, predicted_softmaxs = torch.max(predicted_softmaxs, 1)
				n_class_corrected += (predicted_softmaxs == labels).sum().item()
				total_samples += len(samples)
				self.calibrate_optimizer.zero_grad()
				classification_loss.backward()
				self.calibrate_optimizer.step()
				self.calibrate_scheduler.step()
				if iteration % self.args.step_eval == (self.args.step_eval - 1):
					self.writer.add_scalar("Accuracy/train", 100.0 * n_class_corrected / total_samples, iteration)
					self.writer.add_scalar("Loss/train", total_classification_loss / self.args.step_eval, iteration)
					logging.info(
						"Train set: Iteration: [{}/{}]\tAccuracy: {}/{} ({:.2f}%)\tLoss: {:.6f}".format(
							iteration + 1,
							self.args.iterations,
							n_class_corrected,
							total_samples,
							100.0 * n_class_corrected / total_samples,
							total_classification_loss / self.args.step_eval,
						)
					)
					self.evaluate(iteration, self.flows)
					n_class_corrected, total_classification_loss, total_samples = 0, 0, 0

	def estimate_density(self):
		checkpoint = torch.load(self.checkpoint_name + ".pt")
		self.model.load_state_dict(checkpoint["model_state_dict"])
		self.model.eval()
		self.flows.train()
		total_estimation_loss = 0
		self.train_iter_loader = iter(self.train_loader)
		for iteration in range(self.args.density_estimation_iterations):
			if (iteration % len(self.train_iter_loader)) == 0:
				self.train_iter_loader = iter(self.train_loader)
			samples, labels = self.train_iter_loader.next()
			samples, labels = samples.to(self.device), labels.to(self.device)
			latents = self.model(samples)
			density_loss = -self.flows.log_probs(latents.detach(), None).mean()
			total_estimation_loss += density_loss
			self.density_optimizer.zero_grad()
			density_loss.backward()
			self.density_optimizer.step()
			if iteration % self.args.step_eval == (self.args.step_eval - 1):
				self.writer.add_scalar("Loss/density", total_estimation_loss / self.args.step_eval, iteration)
				logging.info(
					"Train set: Iteration: [{}/{}]\tLoss: {:.6f}".format(
						iteration + 1,
						self.args.density_estimation_iterations,
						total_estimation_loss / self.args.step_eval,
					)
				)
				total_estimation_loss = 0
		torch.save(
			{
				"score_state_dict": self.flows.state_dict(),
			},
			self.checkpoint_name + "_score.pt",
		)

	def evaluate(self, n_iter, flows = None):
		self.model.eval()
		self.classifier.eval()
		n_class_corrected, total_classification_loss, total_ece = 0, 0, 0
		with torch.no_grad():
			for iteration, (samples, labels) in enumerate(self.val_loader):
				samples, labels = samples.to(self.device), labels.to(self.device)
				latents = self.model(samples)
				predicted_classes = self.classifier(latents)
				if flows is None:
					predicted_softmaxs = self.nn_logsoftmax(predicted_classes)
				else:
					prob = torch.sigmoid(torch.exp(self.flows.log_probs(latents)))
					predicted_softmaxs = self.nn_logsoftmax(prob * predicted_classes)
				total_ece += multiclass_calibration_error(torch.exp(predicted_softmaxs), labels, num_classes=10, n_bins=10, norm='l1')
				classification_loss = self.criterion(predicted_softmaxs, labels)
				total_classification_loss += classification_loss.item()
				_, predicted_softmaxs = torch.max(predicted_softmaxs, 1)
				n_class_corrected += (predicted_softmaxs == labels).sum().item()
		self.writer.add_scalar("Accuracy/validate", 100.0 * n_class_corrected / len(self.val_loader.dataset), n_iter)
		self.writer.add_scalar("Loss/validate", total_classification_loss / len(self.val_loader), n_iter)
		logging.info(
			"Val set: Accuracy: {}/{} ({:.2f}%)\tLoss: {:.6f}\tECE: {:.6f}".format(
				n_class_corrected,
				len(self.val_loader.dataset),
				100.0 * n_class_corrected / len(self.val_loader.dataset),
				total_classification_loss / len(self.val_loader),
				total_ece / len(self.val_loader),
			)
		)
		val_acc = n_class_corrected / len(self.val_loader.dataset)
		val_loss = total_classification_loss / len(self.val_loader)
		self.model.train()
		self.classifier.train()
		if self.args.val_size != 0:
			if self.val_loss_min > val_loss:
				self.val_loss_min = val_loss
				torch.save(
					{
						"model_state_dict": self.model.state_dict(),
						"classifier_state_dict": self.classifier.state_dict(),
					},
					self.checkpoint_name + ".pt",
				)
		else:
			if self.val_acc_max < val_acc:
				self.val_acc_max = val_acc
				torch.save(
					{
						"model_state_dict": self.model.state_dict(),
						"classifier_state_dict": self.classifier.state_dict(),
					},
					self.checkpoint_name + ".pt",
				)

	def test(self):
		self.wb = Workbook()
		sheet = self.wb.add_sheet("test_path")
		checkpoint_score = torch.load(self.checkpoint_name + "_score.pt")
		test_sample_paths, test_class_labels = set_test_samples_labels(self.args.test_meta_filenames)
		checkpoint = torch.load(self.checkpoint_name + ".pt")
		self.model.load_state_dict(checkpoint["model_state_dict"])
		self.classifier.load_state_dict(checkpoint["classifier_state_dict"])
		self.flows.load_state_dict(checkpoint_score["score_state_dict"])
		self.model.eval()
		self.classifier.eval()
		self.flows.eval()
		for i in range(len(self.args.test_paths)):
			test_path = self.args.test_paths[i]
			self.test_loader = DataLoader(
				dataloader_factory.get_test_dataloader(self.args.dataset)(
					path=test_path, sample_paths=test_sample_paths, class_labels=test_class_labels
				),
				batch_size=self.args.batch_size,
				shuffle=True,
			)
			# outputs, outlabels = [], []
			n_class_corrected, total_ece, total_ece_2, total_nll = 0, 0, 0, 0
			with torch.no_grad():
				for iteration, (samples, labels) in enumerate(self.test_loader):
					samples, labels = samples.to(self.device), labels.to(self.device)
					latents = self.model(samples)
					predicted_classes = self.classifier(latents)
					prob = torch.sigmoid(torch.exp(self.flows.log_probs(latents)))
					# outputs += (predicted_classes).tolist()
					# outlabels += labels.tolist()
					predicted_softmaxs = self.nn_logsoftmax(prob * predicted_classes)
					total_ece += multiclass_calibration_error(torch.exp(predicted_softmaxs), labels, num_classes=10, n_bins=10, norm='l1')
					total_nll += self.criterion(predicted_softmaxs, labels).item()

					predicted_softmaxs_2 = self.nn_logsoftmax(predicted_classes)
					total_ece_2 += multiclass_calibration_error(torch.exp(predicted_softmaxs_2), labels, num_classes=10, n_bins=10, norm='l1')

					_, predicted_softmaxs = torch.max(predicted_softmaxs, 1)
					n_class_corrected += (predicted_softmaxs == labels).sum().item()
			test_acc =  100.0 * n_class_corrected / len(self.test_loader.dataset)
			test_ece = round(total_ece.cpu().detach().numpy() / len(self.test_loader), 6)
			total_ece_2 = round(total_ece_2.cpu().detach().numpy() / len(self.test_loader), 6)
			test_nll = round(total_nll / len(self.test_loader), 6)
			print(
				test_path + "\tTest set: Accuracy: {}/{} ({:.2f}%)\tECE: {:.6f}\tECE2: {:.6f}\tNLL: {:.6f}".format(
					n_class_corrected,
					len(self.test_loader.dataset),
					test_acc,
					test_ece,
					total_ece_2,
					test_nll,
				)
			)
			sheet.write(0, i, test_acc)
			sheet.write(1, i, test_ece)
			sheet.write(2, i, test_nll)
			sheet.write(3, i, total_ece_2)
			# outputs = torch.Tensor(outputs)
			# outlabels = torch.Tensor(outlabels)
			# with open("outputs2.pkl", "wb") as fp:
			# 	pickle.dump(outputs, fp)
			# with open("labels.pkl", "wb") as fp:
			# 	pickle.dump(outlabels, fp)
			# print(outputs.shape)
			# quit()
		self.wb.save(self.args.algorithm + "_" + self.args.exp_name + "_" + self.exp_idx + '.xls')

	def save_plot(self):
		checkpoint = torch.load(self.checkpoint_name + ".pt")
		checkpoint_score = torch.load(self.checkpoint_name + "_score.pt")
		self.model.load_state_dict(checkpoint["model_state_dict"])
		self.classifier.load_state_dict(checkpoint["classifier_state_dict"])
		self.flows.load_state_dict(checkpoint_score["score_state_dict"])
		self.model.eval()
		self.classifier.eval()
		self.flows.eval()
		Z_train, Y_train, Z_test, Y_test = [], [], [], []
		tr_nlls, tr_entropies, te_nlls, te_entropies = [], [], [], []
		with torch.no_grad():
			for iteration, (samples, labels) in enumerate(self.train_loader):
				b, c, h, w = samples.shape
				samples, labels = samples.to(self.device), labels.to(self.device)
				z = self.model(samples)
				z_s = z
				predicted_classes = self.classifier(z)
				predicted_softmaxs = self.nn_logsoftmax(predicted_classes)
				for predicted_softmax in predicted_softmaxs:
					tr_entropies.append(entropy(torch.exp(predicted_softmax).cpu().detach().numpy()))
				classification_loss = self.criterion(predicted_softmaxs, labels)
				log_prob = torch.sigmoid(torch.exp(self.flows.log_probs(z)))
				bpd = (log_prob.cpu().detach().sum()) / b
				tr_nlls.append(bpd.numpy())
				Z_train += z.tolist()
				Y_train += labels.tolist()
			for iteration, (samples, labels) in enumerate(self.test_loader):
				b, c, h, w = samples.shape
				samples, labels = samples.to(self.device), labels.to(self.device)
				z = self.model(samples)
				z_s = z
				predicted_classes = self.classifier(z)
				prob = torch.sigmoid(torch.exp(self.flows.log_probs(z)))
				bpd = (prob.cpu().detach().sum()) / b
				predicted_softmaxs = self.nn_logsoftmax(prob * predicted_classes)
				for predicted_softmax in predicted_softmaxs:
					te_entropies.append(entropy(torch.exp(predicted_softmax).cpu().detach().numpy()))
				classification_loss = self.criterion(predicted_softmaxs, labels)
				te_nlls.append(bpd.numpy())
				Z_test += z.tolist()
				Y_test += labels.tolist()
		if not os.path.exists(self.plot_dir):
			os.mkdir(self.plot_dir)
		with open(self.plot_dir + "Z_train.pkl", "wb") as fp:
			pickle.dump(Z_train, fp)
		with open(self.plot_dir + "Y_train.pkl", "wb") as fp:
			pickle.dump(Y_train, fp)
		with open(self.plot_dir + "Z_test.pkl", "wb") as fp:
			pickle.dump(Z_test, fp)
		with open(self.plot_dir + "Y_test.pkl", "wb") as fp:
			pickle.dump(Y_test, fp)
		with open(self.plot_dir + "tr_nlls.pkl", "wb") as fp:
			pickle.dump(tr_nlls, fp)
		with open(self.plot_dir + "tr_entropies.pkl", "wb") as fp:
			pickle.dump(tr_entropies, fp)
		with open(self.plot_dir + "te_nlls.pkl", "wb") as fp:
			pickle.dump(te_nlls, fp)
		with open(self.plot_dir + "te_entropies.pkl", "wb") as fp:
			pickle.dump(te_entropies, fp)

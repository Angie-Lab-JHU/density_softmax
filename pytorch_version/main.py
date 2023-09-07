import argparse
import json
import os
import random

import numpy as np
import torch
from algorithms.ERM.src.Trainer_ERM import Trainer_ERM
from algorithms.KDE.src.Trainer_KDE import Trainer_KDE
from algorithms.Flows.src.Trainer_Flows import Trainer_Flows
from algorithms.VAE.src.Trainer_VAE import Trainer_VAE


def fix_random_seed(seed_value):
	random.seed(seed_value)
	np.random.seed(seed_value)
	torch.manual_seed(seed_value)

	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed_value)
		torch.cuda.manual_seed(seed_value)
		torch.backends.cudnn.enabled = False
		torch.backends.cudnn.benchmark = False
		torch.backends.cudnn.deterministic = True 


algorithms_map = {"ERM": Trainer_ERM, "KDE": Trainer_KDE, "VAE": Trainer_VAE, "Flows": Trainer_Flows}

if __name__ == "__main__":
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument("--config", help="Path to configuration file")
	parser.add_argument("--exp_idx", help="Index of experiment")
	parser.add_argument("--gpu_idx", help="Index of GPU")
	bash_args = parser.parse_args()
	with open(bash_args.config, "r") as inp:
		args = argparse.Namespace(**json.load(inp))
	os.environ["CUDA_VISIBLE_DEVICES"] = bash_args.gpu_idx
	# fix_random_seed(args.seed_value)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	trainer = algorithms_map[args.algorithm](args, device, bash_args.exp_idx)
	trainer.train()
	trainer.test()
	trainer.save_plot()
	print("Finished!")

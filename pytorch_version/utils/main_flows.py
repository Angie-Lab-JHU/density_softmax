import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision


def save_images(epoch, best_model):
    best_model.eval()
    # fixed_noise = torch.Tensor(100, 32).normal_()
    y = torch.arange(1000).unsqueeze(-1) % 10
    y_onehot = torch.FloatTensor(1000, 10)
    y_onehot.zero_()
    y_onehot.scatter_(1, y, 1)

    Z_out, Y_out = [], []

    with torch.no_grad():
        imgs = best_model.sample(1000).detach().cpu()
        Z_out += imgs.tolist()
        Y_out += y.tolist()

    with open("out/Z_Flows.pkl", "wb") as fp:
        pickle.dump(Z_out, fp)

    # with open("Y_out.pkl", "wb") as fp:
    # 	pickle.dump(Y_out, fp)


import argparse
import copy
import math
import pickle
import sys

import flows as fnn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from tqdm import tqdm


# from tensorboardX import SummaryWriter


if sys.version_info < (3, 6):
    print("Sorry, this code might need Python 3.6 or higher")

# Training settings
parser = argparse.ArgumentParser(description="PyTorch Flows")
parser.add_argument("--batch-size", type=int, default=100, help="input batch size for training (default: 100)")
parser.add_argument("--test-batch-size", type=int, default=1000, help="input batch size for testing (default: 1000)")
parser.add_argument("--epochs", type=int, default=1000, help="number of epochs to train (default: 1000)")
parser.add_argument("--lr", type=float, default=0.0001, help="learning rate (default: 0.0001)")
parser.add_argument("--dataset", default="POWER", help="POWER | GAS | HEPMASS | MINIBONE | BSDS300 | MOONS")
parser.add_argument("--flow", default="realnvp", help="flow to use: maf | realnvp | glow")
parser.add_argument("--no-cuda", action="store_true", default=False, help="disables CUDA training")
parser.add_argument("--cond", action="store_true", default=False, help="train class conditional flow (only for MNIST)")
parser.add_argument("--num-blocks", type=int, default=5, help="number of invertible blocks (default: 5)")
parser.add_argument("--seed", type=int, default=1, help="random seed (default: 1)")
parser.add_argument(
    "--log-interval", type=int, default=1000, help="how many batches to wait before logging training status"
)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda:0" if args.cuda else "cpu")

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {} if args.cuda else {}

assert args.dataset in ["POWER", "GAS", "HEPMASS", "MINIBONE", "BSDS300", "MOONS", "MNIST"]

if args.cond:
    # assert args.flow in ['maf', 'realnvp'] and args.dataset == 'MNIST', \
    # 	'Conditional flows are implemented only for maf and MNIST'

    with open("../algorithms/SM/results/plots/MNIST_0/Z_out.pkl", "rb") as fp:
        Z_out = pickle.load(fp)
    with open("../algorithms/SM/results/plots/MNIST_0/Y_out.pkl", "rb") as fp:
        Y_out = pickle.load(fp)

    trn_x = np.asarray(Z_out)
    val_x = np.asarray(Z_out)
    tst_x = np.asarray(Z_out)
    Y_out = np.asarray(Y_out).reshape(len(Y_out), 1)

    trn_y = np.zeros((Y_out.size, Y_out.max() + 1))
    trn_y[np.arange(Y_out.size), Y_out] = 1

    train_tensor = torch.from_numpy(trn_x).float()
    train_labels = torch.from_numpy(trn_y).float()
    train_dataset = torch.utils.data.TensorDataset(train_tensor, train_labels)

    valid_tensor = torch.from_numpy(val_x).float()
    valid_labels = torch.from_numpy(trn_y).float()
    valid_dataset = torch.utils.data.TensorDataset(valid_tensor, valid_labels)

    test_tensor = torch.from_numpy(tst_x).float()
    test_labels = torch.from_numpy(trn_y).float()
    test_dataset = torch.utils.data.TensorDataset(test_tensor, test_labels)
    num_cond_inputs = 10
else:
    with open("../algorithms/SM/results/plots/MNIST_0/Z_out.pkl", "rb") as fp:
        Z_out = pickle.load(fp)

    trn_x = np.asarray(Z_out)
    val_x = np.asarray(Z_out)
    tst_x = np.asarray(Z_out)

    train_tensor = torch.from_numpy(trn_x).float()
    train_dataset = torch.utils.data.TensorDataset(train_tensor)

    valid_tensor = torch.from_numpy(val_x).float()
    valid_dataset = torch.utils.data.TensorDataset(valid_tensor)

    test_tensor = torch.from_numpy(tst_x).float()
    test_dataset = torch.utils.data.TensorDataset(test_tensor)
    num_cond_inputs = None

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

valid_loader = torch.utils.data.DataLoader(
    valid_dataset, batch_size=args.test_batch_size, shuffle=False, drop_last=False, **kwargs
)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=args.test_batch_size, shuffle=False, drop_last=False, **kwargs
)

num_inputs = 32
num_hidden = {"POWER": 100, "GAS": 100, "HEPMASS": 512, "MINIBOONE": 512, "BSDS300": 512, "MOONS": 64, "MNIST": 1024}[
    args.dataset
]

act = "relu"

modules = []

assert args.flow in ["maf", "maf-split", "maf-split-glow", "realnvp", "glow"]
if args.flow == "glow":
    mask = torch.arange(0, num_inputs) % 2
    mask = mask.to(device).float()

    print("Warning: Results for GLOW are not as good as for MAF yet.")
    for _ in range(args.num_blocks):
        modules += [
            fnn.BatchNormFlow(num_inputs),
            fnn.LUInvertibleMM(num_inputs),
            fnn.CouplingLayer(num_inputs, num_hidden, mask, num_cond_inputs, s_act="tanh", t_act="relu"),
        ]
        mask = 1 - mask
elif args.flow == "realnvp":
    mask = torch.arange(0, num_inputs) % 2
    mask = mask.to(device).float()

    for _ in range(args.num_blocks):
        modules += [
            fnn.CouplingLayer(num_inputs, num_hidden, mask, num_cond_inputs, s_act="tanh", t_act="relu"),
            fnn.BatchNormFlow(num_inputs),
        ]
        mask = 1 - mask
elif args.flow == "maf":
    for _ in range(args.num_blocks):
        modules += [
            fnn.MADE(num_inputs, num_hidden, num_cond_inputs, act=act),
            fnn.BatchNormFlow(num_inputs),
            fnn.Reverse(num_inputs),
        ]
elif args.flow == "maf-split":
    for _ in range(args.num_blocks):
        modules += [
            fnn.MADESplit(num_inputs, num_hidden, num_cond_inputs, s_act="tanh", t_act="relu"),
            fnn.BatchNormFlow(num_inputs),
            fnn.Reverse(num_inputs),
        ]
elif args.flow == "maf-split-glow":
    for _ in range(args.num_blocks):
        modules += [
            fnn.MADESplit(num_inputs, num_hidden, num_cond_inputs, s_act="tanh", t_act="relu"),
            fnn.BatchNormFlow(num_inputs),
            fnn.InvertibleMM(num_inputs),
        ]

model = fnn.FlowSequential(*modules)

for module in model.modules():
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight)
        if hasattr(module, "bias") and module.bias is not None:
            module.bias.data.fill_(0)

model.to(device)

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)

# writer = SummaryWriter(comment=args.flow + "_" + args.dataset)
global_step = 0


def train(epoch):
    global global_step
    model.train()
    train_loss = 0

    pbar = tqdm(total=len(train_loader.dataset))
    for batch_idx, data in enumerate(train_loader):
        if isinstance(data, list):
            if len(data) > 1:
                cond_data = data[1].float()
                cond_data = cond_data.to(device)
            else:
                cond_data = None

            data = data[0]
        data = data.to(device)
        optimizer.zero_grad()
        loss = -model.log_probs(data, cond_data).mean()
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

        pbar.update(data.size(0))
        pbar.set_description("Train, Log likelihood in nats: {:.6f}".format(-train_loss / (batch_idx + 1)))

        print("training/loss", loss.item(), global_step)
        global_step += 1

    pbar.close()

    for module in model.modules():
        if isinstance(module, fnn.BatchNormFlow):
            module.momentum = 0

    if args.cond:
        with torch.no_grad():
            model(
                train_loader.dataset.tensors[0].to(data.device),
                train_loader.dataset.tensors[1].to(data.device).float(),
            )
    else:
        with torch.no_grad():
            model(train_loader.dataset.tensors[0].to(data.device))

    for module in model.modules():
        if isinstance(module, fnn.BatchNormFlow):
            module.momentum = 1


def validate(epoch, model, loader, prefix="Validation"):
    global global_step

    model.eval()
    val_loss = 0

    pbar = tqdm(total=len(loader.dataset))
    pbar.set_description("Eval")
    for batch_idx, data in enumerate(loader):
        if isinstance(data, list):
            if len(data) > 1:
                cond_data = data[1].float()
                cond_data = cond_data.to(device)
            else:
                cond_data = None

            data = data[0]
        data = data.to(device)
        with torch.no_grad():
            val_loss += -model.log_probs(data, cond_data).sum().item()  # sum up batch loss
        pbar.update(data.size(0))
        pbar.set_description("Val, Log likelihood in nats: {:.6f}".format(-val_loss / pbar.n))

    print("validation/LL", val_loss / len(loader.dataset), epoch)

    pbar.close()
    return val_loss / len(loader.dataset)


best_validation_loss = float("inf")
best_validation_epoch = 0
best_model = model

for epoch in range(args.epochs):
    print("\nEpoch: {}".format(epoch))

    train(epoch)
    validation_loss = validate(epoch, model, valid_loader)

    if epoch - best_validation_epoch >= 30:
        break

    if validation_loss < best_validation_loss:
        best_validation_epoch = epoch
        best_validation_loss = validation_loss
        best_model = copy.deepcopy(model)

    print(
        "Best validation at epoch {}: Average Log Likelihood in nats: {:.4f}".format(
            best_validation_epoch, -best_validation_loss
        )
    )

    # if args.dataset == 'MNIST' and epoch % 1 == 0:
    save_images(epoch, model)


validate(best_validation_epoch, best_model, test_loader, prefix="Test")

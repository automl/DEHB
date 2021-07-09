"""
This script runs a Hyperparameter Optimisation (HPO) using DEHB to tune the architecture and
training hyperparameters for training a neural network on MNIST in PyTorch.

The parameter space is defined in the get_configspace() function. Any configuration sampled from
this space can be passed to an object of class Model() which can instantiate a CNN architecture
from it. The objective_function() is the target function that DEHB minimizes for this problem. This
function instantiates an architecture, an optimizer, as defined by a configuration and performs the
training and evaluation (on the validation set) as per the budget passed.
The argument `runtime` can be passed to DEHB as a wallclock budget for running the optimisation.

This tutorial also briefly refers to the different methods of interfacing DEHB with the Dask
parallelism framework. Moreover, also introduce how GPUs may be managed, which is recommended for
running this example tutorial.

Additional requirements:
* torch>=1.7.1
* torchvision>=0.8.2
* torchsummary>=1.5.1

PyTorch code referenced from: https://github.com/pytorch/examples/blob/master/mnist/main.py
"""


import os
import time
import pickle
import argparse
import numpy as np
from distributed import Client

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torchsummary import summary

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from dehb import DEHB


class Model(nn.Module):
    def __init__(self, config, img_dim=28, output_dim=10):
        super().__init__()
        self.output_dim = output_dim
        self.pool_kernel = 2
        self.pool_stride = 1
        self.maxpool = nn.MaxPool2d(self.pool_kernel, self.pool_stride)
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=config["channels_1"],
            kernel_size=config["kernel_1"],
            stride=config["stride_1"],
            padding=0,
            dilation=1
        )
        # updating image size after conv1
        img_dim = self._update_size(img_dim, config["kernel_1"], config["stride_1"], 0, 1)
        self.conv2 = nn.Conv2d(
            in_channels=config["channels_1"],
            out_channels=config["channels_2"],
            kernel_size=config["kernel_2"],
            stride=config["stride_2"],
            padding=0,
            dilation=1
        )
        # updating image size after conv2
        img_dim = self._update_size(img_dim, config["kernel_2"], config["stride_2"], 0, 1)
        # updating image size after maxpool
        img_dim = self._update_size(img_dim, self.pool_kernel, self.pool_stride, 0, 1)
        self.dropout = nn.Dropout(config["dropout"])
        hidden_dim = config["hidden"]
        self.fc1 = nn.Linear(img_dim * img_dim * config["channels_2"], hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, self.output_dim)

    def forward(self, x):
        # Layer 1
        x = self.conv1(x)
        x = F.relu(x)
        x = self.dropout(x)
        # Layer 2
        x = self.conv2(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = self.dropout(x)
        # FC Layer 1
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        # Output layer
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

    def _update_size(self, dim, kernel_size, stride, padding, dilation):
        return int(np.floor((dim + 2 * padding - (dilation * (kernel_size - 1) + 1)) / stride + 1))


def get_configspace(seed=None):
    cs = CS.ConfigurationSpace(seed)

    # Hyperparameter defining first Conv layer
    kernel1 = CSH.OrdinalHyperparameter("kernel_1", sequence=[3, 5, 7], default_value=5)
    channels1 = CSH.UniformIntegerHyperparameter("channels_1", lower=3, upper=64,
                                                 default_value=32)
    stride1 = CSH.UniformIntegerHyperparameter("stride_1", lower=1, upper=2, default_value=1)
    cs.add_hyperparameters([kernel1, channels1, stride1])

    # Hyperparameter defining second Conv layer
    kernel2 = CSH.OrdinalHyperparameter("kernel_2", sequence=[3, 5, 7], default_value=5)
    channels2 = CSH.UniformIntegerHyperparameter("channels_2", lower=3, upper=64,
                                                 default_value=32)
    stride2 = CSH.UniformIntegerHyperparameter("stride_2", lower=1, upper=2, default_value=1)
    cs.add_hyperparameters([kernel2, channels2, stride2])

    # Hyperparameter for FC layer
    hidden = CSH.UniformIntegerHyperparameter(
        "hidden", lower=32, upper=256, log=True, default_value=128
    )
    cs.add_hyperparameter(hidden)

    # Regularization Hyperparameter
    dropout = CSH.UniformFloatHyperparameter("dropout", lower=0, upper=0.5, default_value=0.1)
    cs.add_hyperparameter(dropout)

    # Training Hyperparameters
    batch_size = CSH.OrdinalHyperparameter(
        "batch_size", sequence=[2, 4, 8, 16, 32, 64], default_value=4
    )
    lr = CSH.UniformFloatHyperparameter("lr", lower=1e-6, upper=0.1, log=True,
                                        default_value=1e-3)
    cs.add_hyperparameters([batch_size, lr])
    return cs


def train(model, device, train_loader, optimizer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()


def evaluate(model, device, data_loader, acc=False):
    model.eval()
    loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    loss /= len(data_loader.dataset)
    correct /= len(data_loader.dataset)

    if acc:
        return correct
    return loss


def train_and_evaluate(config, max_budget, verbose=False, **kwargs):
    device = kwargs["device"]
    batch_size = config["batch_size"]
    train_set = kwargs["train_set"]
    test_set = kwargs["test_set"]
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    model = Model(config).to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=config["lr"])
    for epoch in range(1, int(max_budget)+1):
        train(model, device, train_loader, optimizer)
    accuracy = evaluate(model, device, test_loader, acc=True)
    if verbose:
        summary(model, (1, 28, 28))  # image dimensions for MNIST
    return accuracy


def objective_function(config, budget, **kwargs):
    """ The target function to minimize for HPO"""
    device = kwargs["device"]

    # Data Loaders
    batch_size = config["batch_size"]
    train_set = kwargs["train_set"]
    valid_set = kwargs["valid_set"]
    test_set = kwargs["test_set"]
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # Build model
    model = Model(config).to(device)

    # Optimizer
    optimizer = optim.Adadelta(model.parameters(), lr=config["lr"])

    start = time.time()  # measuring wallclock time
    for epoch in range(1, int(budget)+1):
        train(model, device, train_loader, optimizer)
    loss = evaluate(model, device, valid_loader)
    cost = time.time() - start

    # not including test score computation in the `cost`
    test_loss = evaluate(model, device, test_loader)

    # dict representation that DEHB requires
    res = {
        "fitness": loss,
        "cost": cost,
        "info": {"test_loss": test_loss, "budget": budget}
    }
    return res


def input_arguments():
    parser = argparse.ArgumentParser(description='Optimizing MNIST in PyTorch using DEHB.')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=123, metavar='S',
                        help='random seed (default: 123)')
    parser.add_argument('--refit_training', action='store_true', default=False,
                        help='Refit with incumbent configuration on full training data and budget')
    parser.add_argument('--min_budget', type=float, default=None,
                        help='Minimum budget (epoch length)')
    parser.add_argument('--max_budget', type=float, default=None,
                        help='Maximum budget (epoch length)')
    parser.add_argument('--eta', type=int, default=3,
                        help='Parameter for Hyperband controlling early stopping aggressiveness')
    parser.add_argument('--output_path', type=str, default="./pytorch_mnist_dehb",
                        help='Directory for DEHB to write logs and outputs')
    parser.add_argument('--scheduler_file', type=str, default=None,
                        help='The file to connect a Dask client with a Dask scheduler')
    parser.add_argument('--n_workers', type=int, default=1,
                        help='Number of CPU workers for DEHB to distribute function evaluations to')
    parser.add_argument('--single_node_with_gpus', default=False, action="store_true",
                        help='If True, signals the DEHB run to assume all required GPUs are on '
                             'the same node/machine. To be specified as True if no client is '
                             'passed and n_workers > 1. Should be set to False if a client is '
                             'specified as a scheduler-file created. The onus of GPU usage is then'
                             'on the Dask workers created and mapped to the scheduler-file.')
    parser.add_argument('--verbose', action="store_true", default=False,
                        help='Decides verbosity of DEHB optimization')
    parser.add_argument('--runtime', type=float, default=300,
                        help='Total time in seconds as budget to run DEHB')
    args = parser.parse_args()
    return args


def main():
    args = input_arguments()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    torch.manual_seed(args.seed)

    # Data Preparation
    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_set = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    train_set, valid_set = torch.utils.data.random_split(train_set, [50000, 10000])
    test_set = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )

    # Get configuration space
    cs = get_configspace(args.seed)
    dimensions = len(cs.get_hyperparameters())

    # Some insights into Dask interfaces to DEHB and handling GPU devices for parallelism:
    # * if args.scheduler_file is specified, args.n_workers need not be specifed --- since
    #    args.scheduler_file indicates a Dask client/server is active
    # * if args.scheduler_file is not specified and args.n_workers > 1 --- the DEHB object
    #    creates a Dask client as at instantiation and dies with the associated DEHB object
    # * if args.single_node_with_gpus is True --- assumes that all GPU devices indicated
    #    through the environment variable "CUDA_VISIBLE_DEVICES" resides on the same machine

    # Dask checks and setups
    single_node_with_gpus = args.single_node_with_gpus
    if args.scheduler_file is not None and os.path.isfile(args.scheduler_file):
        client = Client(scheduler_file=args.scheduler_file)
        # explicitly delegating GPU handling to Dask workers defined
        single_node_with_gpus = False
    else:
        client = None

    ###########################
    # DEHB optimisation block #
    ###########################
    np.random.seed(args.seed)
    dehb = DEHB(f=objective_function, cs=cs, dimensions=dimensions, min_budget=args.min_budget,
                max_budget=args.max_budget, eta=args.eta, output_path=args.output_path,
                # if client is not None and of type Client, n_workers is ignored
                # if client is None, a Dask client with n_workers is set up
                client=client, n_workers=args.n_workers)
    traj, runtime, history = dehb.run(total_cost=args.runtime, verbose=args.verbose, 
                                      # arguments below are part of **kwargs shared across workers
                                      train_set=train_set, valid_set=valid_set, test_set=test_set,
                                      single_node_with_gpus=single_node_with_gpus, device=device)
    # end of DEHB optimisation

    # Saving optimisation trace history
    name = time.strftime("%x %X %Z", time.localtime(dehb.start))
    name = name.replace("/", '-').replace(":", '-').replace(" ", '_')
    dehb.logger.info("Saving optimisation trace history...")
    with open(os.path.join(args.output_path, "history_{}.pkl".format(name)), "wb") as f:
        pickle.dump(history, f)

    # Retrain and evaluate best found configuration
    if args.refit_training:
        dehb.logger.info("Retraining on complete training data to compute test metrics...")
        train_set = torchvision.datasets.MNIST(
            root='./data', train=True, download=True, transform=transform
        )
        incumbent = dehb.vector_to_configspace(dehb.inc_config)
        acc = train_and_evaluate(incumbent, args.max_budget, verbose=True,
                                 train_set=train_set, test_set=test_set, device=device)
        dehb.logger.info("Test accuracy of {:.3f} for the best found configuration: ".format(acc))
        dehb.logger.info(incumbent)


if __name__ == "__main__":
    main()

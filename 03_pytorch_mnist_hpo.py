import time
import argparse
import numpy as np
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
        # x = F.max_pool2d(x, self.pool_kernel, self.pool_stride)
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
    device = kwargs["device"]

    # Data Loaders
    batch_size = config["batch_size"]
    train_set = kwargs["train_set"]
    valid_set = kwargs["valid_set"]
    test_set = kwargs["test_set"]
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=False)

    # Build model
    model = Model(config).to(device)

    start = time.time()  # measuring wallclock time

    # Optimizer
    optimizer = optim.Adadelta(model.parameters(), lr=config["lr"])
    for epoch in range(1, int(budget)+1):
        train(model, device, train_loader, optimizer)

    loss = evaluate(model, device, valid_loader)
    cost = time.time() - start

    return loss, cost


def main():
    parser = argparse.ArgumentParser(description='Optimizing MNIST in PyTorch using DEHB.')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=123, metavar='S',
                        help='random seed (default: 123)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--refit-save', action='store_true', default=False,
                        help='Refit with incumbent on full budget and save model')
    parser.add_argument('--min_budget', type=float, default=None,
                        help='Minimum budget (epoch length)')
    parser.add_argument('--max_budget', type=float, default=None,
                        help='Maximum budget (epoch length)')
    parser.add_argument('--eta', type=int, default=3,
                        help='Parameter for Hyperband controlling early stopping aggressiveness')
    parser.add_argument('--output_path', type=str, default="./pytorch_mnist_dehb",
                        help='Directory for DEHB to write logs and outputs')
    parser.add_argument('--n_workers', type=int, default=1,
                        help='Number of CPU workers for DEHB to distribute function evaluations to')
    parser.add_argument('--verbose', action="store_true", default=False,
                        help='Decides verbosity of DEHB optimization')
    parser.add_argument('--runtime', type=float, default=300,
                        help='Total time in seconds as budget to run DEHB')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    # Data Preparation
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
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

    # DEHB optimization
    np.random.seed(args.seed)
    dehb = DEHB(f=objective_function, cs=cs, dimensions=dimensions, min_budget=args.min_budget,
                max_budget=args.max_budget, eta=args.eta, output_path=args.output_path)
    traj, runtime, history = dehb.run(total_cost=args.runtime, verbose=args.verbose, device=device,
                                      train_set=train_set, valid_set=valid_set, test_set=test_set)

    # Retrain and evaluate best found configuration
    train_set = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    incumbent = dehb.vector_to_configspace(dehb.inc_config)
    acc = train_and_evaluate(incumbent, args.max_budget, verbose=True,
                             train_set=train_set, test_set=test_set, device=device)
    print("Test accuracy of {:.3f} for the best found configuration: ".format(acc))
    print(incumbent)


if __name__ == "__main__":
    main()

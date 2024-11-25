import random
from typing import List, NamedTuple, Tuple

import numpy as np
import torch
import torchvision
from torchvision import transforms
from tqdm import tqdm

# Seed all random number generators
np.random.seed(197331)
torch.manual_seed(197331)
random.seed(197331)


class NetworkConfiguration(NamedTuple):
    n_channels: Tuple[int, ...] = (16, 32, 48)
    kernel_sizes: Tuple[int, ...] = (3, 3, 3)
    strides: Tuple[int, ...] = (1, 1, 1)
    dense_hiddens: Tuple[int, ...] = (256, 256)


class Trainer:
    def __init__(
        self,
        network_type: str = "mlp",
        net_config: NetworkConfiguration = NetworkConfiguration(),
        lr: float = 0.001,
        batch_size: int = 128,
        activation_name: str = "relu",
    ):
        self.lr = lr
        self.batch_size = batch_size
        self.train, self.test = self.load_dataset(self)
        dataiter = iter(self.train)
        images, labels = next(dataiter)
        input_dim = images.shape[1:]
        self.network_type = network_type
        activation_function = self.create_activation_function(activation_name)
        if network_type == "mlp":
            self.network = self.create_mlp(
                input_dim[0] * input_dim[1] * input_dim[2],
                net_config,
                activation_function,
            )
        elif network_type == "cnn":
            self.network = self.create_cnn(
                input_dim[0], net_config, activation_function
            )
        else:
            raise ValueError("Network type not supported")
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)

        self.train_logs = {
            "train_loss": [],
            "test_loss": [],
            "train_mae": [],
            "test_mae": [],
        }

    @staticmethod
    def load_dataset(self):
        transform = transforms.ToTensor()

        trainset = torchvision.datasets.SVHN(
            root="./data", split="train", download=True, transform=transform
        )
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=self.batch_size, shuffle=True
        )

        testset = torchvision.datasets.SVHN(
            root="./data", split="test", download=True, transform=transform
        )
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=self.batch_size, shuffle=False
        )

        return trainloader, testloader

    @staticmethod
    def create_mlp(
        input_dim: int, net_config: NetworkConfiguration, activation: torch.nn.Module
    ) -> torch.nn.Module:
        """
        Create a multi-layer perceptron (MLP) network.

        :param net_config: a NetworkConfiguration named tuple. Only the field 'dense_hiddens' will be used.
        :param activation: The activation function to use.
        :return: A PyTorch model implementing the MLP.
        """
        layers = []
        hidden_dims = net_config.dense_hiddens
        layers.append(torch.nn.Linear(input_dim, hidden_dims[0]))
        layers.append(activation)
        for i in range(1, len(hidden_dims)):
            layers.append(torch.nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
            layers.append(activation)
        layers.append(torch.nn.Linear(hidden_dims[-1], 1))
        return torch.nn.Sequential(*layers)

    @staticmethod
    def create_cnn(
        in_channels: int, net_config: NetworkConfiguration, activation: torch.nn.Module
    ) -> torch.nn.Module:
        """
        Create a convolutional network.

        :param in_channels: The number of channels in the input image.
        :param net_config: a NetworkConfiguration specifying the architecture of the CNN.
        :param activation: The activation function to use.
        :return: A PyTorch model implementing the CNN.
        """
        layers = []
        n_channels = net_config.n_channels
        kernel_sizes = net_config.kernel_sizes
        strides = net_config.strides
        dense_hiddens = net_config.dense_hiddens
        layers.append(
            torch.nn.Conv2d(in_channels, n_channels[0], kernel_sizes[0], strides[0])
        )
        layers.append(activation)
        for i in range(1, len(net_config.n_channels) - 1):
            layers.append(
                torch.nn.Conv2d(
                    n_channels[i - 1], n_channels[i], kernel_sizes[i], strides[i]
                )
            )
            layers.append(activation)
            layers.append(torch.nn.MaxPool2d(kernel_size=2))
        layers.append(
            torch.nn.Conv2d(
                n_channels[-2], n_channels[-1], kernel_sizes[-1], strides[-1]
            )
        )
        layers.append(activation)
        layers.append(torch.nn.AdaptiveMaxPool2d((4, 4)))
        layers.append(torch.nn.Flatten())
        layers.append(torch.nn.Linear(n_channels[-1] * 16, dense_hiddens[0]))
        layers.append(activation)
        for i in range(1, len(dense_hiddens)):
            layers.append(torch.nn.Linear(dense_hiddens[i - 1], dense_hiddens[i]))
            layers.append(activation)
        layers.append(torch.nn.Linear(dense_hiddens[-1], 1))
        return torch.nn.Sequential(*layers)

    @staticmethod
    def create_activation_function(activation_str: str) -> torch.nn.Module:
        if activation_str == "relu":
            return torch.nn.ReLU()
        if activation_str == "tanh":
            return torch.nn.Tanh()
        return torch.nn.Sigmoid()

    def compute_loss_and_mae(
        self, X: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.network_type == "mlp":
            X = torch.flatten(X, start_dim=1)
        X = X.to("mps")
        y = y.to("mps")
        preds = self.network(X).flatten()
        mse_loss = torch.nn.MSELoss()
        mae_loss = torch.nn.L1Loss()
        return mse_loss(preds, y), mae_loss(preds, y)

    def training_step(self, X_batch: torch.Tensor, y_batch: torch.Tensor):
        mse_loss, mae_loss = self.compute_loss_and_mae(X_batch, y_batch)
        self.optimizer.zero_grad()
        mse_loss.backward(retain_graph=True)
        mae_loss.backward()
        self.optimizer.step()
        return mse_loss, mae_loss

    def train_loop(self, n_epochs: int) -> dict:
        N = len(self.train)
        for epoch in tqdm(range(n_epochs)):
            train_loss = 0.0
            train_mae = 0.0
            for i, data in enumerate(self.train):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                loss, mae = self.training_step(inputs, labels)
                train_loss += loss
                train_mae += mae

            # Log data every epoch
            self.train_logs["train_mae"].append(train_mae / N)
            self.train_logs["train_loss"].append(train_loss / N)
            self.evaluation_loop()

        return self.train_logs

    def evaluation_loop(self) -> None:
        self.network.eval()
        N = len(self.test)
        with torch.inference_mode():
            test_loss = 0.0
            test_mae = 0.0
            for data in self.test:
                inputs, labels = data
                loss, mae = self.compute_loss_and_mae(inputs, labels)
                test_loss += loss.item()
                test_mae += mae.item()

        self.train_logs["test_mae"].append(test_mae / N)
        self.train_logs["test_loss"].append(test_loss / N)

    def evaluate(
        self, X: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mse_loss, mae_loss = self.compute_loss_and_mae(X, y)
        return torch.sum(mse_loss), torch.sum(mae_loss)


def main():
    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print(
                "MPS not available because the current PyTorch install was not "
                "built with MPS enabled."
            )
        else:
            print(
                "MPS not available because the current MacOS version is not 12.3+ "
                "and/or you do not have an MPS-enabled device on this machine."
            )
    else:
        mps_device = torch.device("mps")
        net_config = NetworkConfiguration(dense_hiddens=(128, 128))
        # learning_rates
        trainer = Trainer(
            network_type="mlp",
            net_config=net_config,
            batch_size=128,
            activation_name="relu",
        )
        trainer.network.to(mps_device)
        train_logs = trainer.train_loop(50)
        print(train_logs)


if __name__ == "__main__":
    main()

import random
import numpy as np
import torch
from typing import Tuple, List, NamedTuple
from tqdm import tqdm
import torchvision
from torchvision import transforms


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
    def __init__(self,
                 network_type: str = "mlp",
                 net_config: NetworkConfiguration = NetworkConfiguration(),
                 lr: float = 0.001,
                 batch_size: int = 128,
                 activation_name: str = "relu"):

        self.lr = lr
        self.batch_size = batch_size
        self.train, self.test = self.load_dataset(self)
        dataiter = iter(self.train)
        images, labels = next(dataiter)
        input_dim = images.shape[1:]
        self.network_type = network_type
        activation_function = self.create_activation_function(activation_name)
        if network_type == "mlp":
            self.network = self.create_mlp(input_dim[0]*input_dim[1]*input_dim[2], 
                                           net_config,
                                           activation_function)
        elif network_type == "cnn":
            self.network = self.create_cnn(input_dim[0], 
                                           net_config, 
                                           activation_function)
        else:
            raise ValueError("Network type not supported")
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)

        self.train_logs = {'train_loss': [], 'test_loss': [],
                           'train_mae': [], 'test_mae': []}

    @staticmethod
    def load_dataset(self):
        transform = transforms.ToTensor()

        trainset = torchvision.datasets.SVHN(root='./data', split='train',
                                                download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size,
                                                shuffle=True)

        testset = torchvision.datasets.SVHN(root='./data', split='test',
                                            download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size,
                                                shuffle=False)

        return trainloader, testloader

    @staticmethod
    def create_mlp(input_dim: int, net_config: NetworkConfiguration,
                   activation: torch.nn.Module) -> torch.nn.Module:
        """
        Create a multi-layer perceptron (MLP) network.

        :param net_config: a NetworkConfiguration named tuple. Only the field 'dense_hiddens' will be used.
        :param activation: The activation function to use.
        :return: A PyTorch model implementing the MLP.
        """
        # TODO write code here
        pass

    @staticmethod
    def create_cnn(in_channels: int, net_config: NetworkConfiguration,
                   activation: torch.nn.Module) -> torch.nn.Module:
        """
        Create a convolutional network.

        :param in_channels: The number of channels in the input image.
        :param net_config: a NetworkConfiguration specifying the architecture of the CNN.
        :param activation: The activation function to use.
        :return: A PyTorch model implementing the CNN.
        """
        # TODO write code here
        pass

    @staticmethod
    def create_activation_function(activation_str: str) -> torch.nn.Module:
        # TODO WRITE CODE HERE
        pass

    def compute_loss_and_mae(self, X: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO WRITE CODE HERE
        pass

    def training_step(self, X_batch: torch.Tensor, y_batch: torch.Tensor):
        # TODO WRITE CODE HERE
        pass

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
            self.train_logs['train_mae'].append(train_mae / N)
            self.train_logs['train_loss'].append(train_loss / N)
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

        self.train_logs['test_mae'].append(test_mae / N)
        self.train_logs['test_loss'].append(test_loss / N)


    def evaluate(self, X: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO WRITE CODE HERE
        pass

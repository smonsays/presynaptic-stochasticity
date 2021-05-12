"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import abc

import torch

from lib import ddc


class NeuralNetwork(torch.nn.Module, abc.ABC):
    """
    Abstract neural network class
    """
    def __init__(self, layers, input_dim, output_dim):
        """
        Args:
            layers: torch.nn.ModuleList containing the layers of the network
            input_dim: Integer with the input dimension of the network
            output_dim: Integer with the output dimension of the network
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        assert isinstance(layers, torch.nn.ModuleList)
        self.layers = layers

    @property
    @abc.abstractmethod
    def energy(self):
        """
        Energy required to sustain the weights in the network.
        """
        raise NotImplementedError


class MultilayerPerceptron(NeuralNetwork):
    """
    Multilayer Perceptron.
    """
    def __init__(self, dimensions, nonlinearity, fixed_bias=False, prob_release=None):
        """
        Args:
            dimensions: List of integers defining the network architecture
            nonlinearity: Function used as nonlinearity between layers
            fixed_bias: Boolean defining whether biases should be fixed
            prob_release: Float defining the fixed release (transmission) probability of the weights
        """
        layers = torch.nn.ModuleList(
            torch.nn.Linear(dim1, dim2, bias=fixed_bias)
            for dim1, dim2 in zip(dimensions[:-1], dimensions[1:])
        )

        if prob_release is not None:
            for module in layers:
                ddc._register_dropconnect_pre_hook(module, prob_release, normalise=True)

        super().__init__(layers, dimensions[0], dimensions[-1])

        self.dimensions = dimensions
        self.nonlinearity = nonlinearity
        self.prob_release = prob_release

    @property
    @torch.no_grad()
    def energy(self):
        return torch.sum(torch.cat([torch.abs(l.weight).view(-1) for l in self.layers]))

    def forward(self, x):
        x = x.view((-1, self.dimensions[0]))

        for l in self.layers[:-1]:
            x = l(x)
            x = self.nonlinearity(x)

        # The output is unnormalized (i.e. no logits or probabilities)
        y_pred = self.layers[-1](x)

        return y_pred

    def activation(self, x):
        """
        Sum over all hidden activations to measure energy for a given input.
        """
        x = x.view((-1, self.dimensions[0]))
        energy = torch.zeros(x.shape[0], device=x.device)

        for l in self.layers[:-1]:
            x = l(x)
            x = self.nonlinearity(x)
            energy += torch.sum(x, dim=1)

        return energy


class SmallCNN(NeuralNetwork):
    def __init__(self):
        layers = torch.nn.ModuleList([
            torch.nn.Conv2d(1, 32, 3, 1),
            torch.nn.Conv2d(32, 64, 3, 1),
            torch.nn.Linear(9216, 128),
            torch.nn.Linear(128, 10),
        ])
        super().__init__(layers, 28, 10)

    def forward(self, x):
        x = self.layers[0](x)  # conv2d
        x = torch.nn.functional.relu(x)
        x = self.layers[1](x)  # conv2d
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.layers[2](x)   # linear
        x = torch.nn.functional.relu(x)
        x = self.layers[3](x)  # linear

        return x


class ZenkeNet(NeuralNetwork):
    def __init__(self, num_datasets=1, num_classes=10):
        self.num_datasets = num_datasets
        self.num_classes = num_classes

        layers = torch.nn.ModuleList([
            torch.nn.Conv2d(3, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(32, 32, kernel_size=3, padding=0),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            # torch.nn.Dropout(p=0.25),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=0),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            # torch.nn.Dropout(p=0.25),
            torch.nn.Flatten(),
            torch.nn.Linear(2304, 512),
            torch.nn.ReLU(inplace=True),
            # torch.nn.Dropout(p=0.5),
            torch.nn.Linear(512, self.num_classes)
        ])

        super().__init__(layers, torch.Size([3, 32, 32]), self.num_classes)

        # self.output_heads = torch.nn.ModuleList(
        #     torch.nn.Linear(512, self.num_classes) for _ in range(self.num_datasets)
        # )

    def forward(self, x, dataset_idx=None):

        for l in self.layers:
            x = l(x)

        # return self.output_heads[dataset_idx](x)
        return x

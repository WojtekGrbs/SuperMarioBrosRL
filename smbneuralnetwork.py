import torch
from torch import nn
import numpy as np
from preparation import STACKED_FRAMES


class SMBNeuralNetwork(nn.Module):
    '''
    Neural Network class allowing an evaluation of a 4-stacked screen of 64x64 pixels into
    a (1,5) array of scores.
    Evaluation parameter differentiates the action network from evaluation network. The latter one is not meant
    to have its parameters updated in backprop. The architecture of both networks needs to remain the same, although
    its structure is not considered optimal. Requires further investigation.
    '''
    def __init__(self, evaluation=False):
        super().__init__()
        self.network = nn.Sequential(
        # First conv layer
        nn.Conv2d(in_channels=STACKED_FRAMES, out_channels=32, kernel_size=8, stride=4),
        nn.ReLU(),
        # Second conv layer
        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1),
        nn.ReLU(),

        nn.Flatten(),
        # First linear layer
        nn.Linear(288, 512),
        # Second linear layer
        nn.ReLU(),

        # 5 values for output layer, same length as RIGHT_ONLY action space
        nn.Linear(512, 5))

        if evaluation:
            for p in self.network.parameters():
                p.requires_grad = False

    def forward(self, input):
        return self.network(input)

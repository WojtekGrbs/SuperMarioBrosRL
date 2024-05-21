import torch
from torch import nn
from preparation import STACKED_FRAMES

CUDA_FLAG = torch.cuda.is_available()
DEVICE = "cuda" if CUDA_FLAG else "cpu"

if CUDA_FLAG:
    print('Wykryto karte, uczenie normalne:')
    print(torch.cuda.get_device_name(0))
else:
    print('Nie wykryto żadnego GPU, uczenie włączone w trybie testowania kodu')

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
        nn.Conv2d(STACKED_FRAMES, 32, kernel_size=8, stride=4),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=4, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 128, kernel_size=2, stride=1),
        nn.ReLU(),
        nn.Flatten(),
        # First linear layer
        nn.Linear(3200, 512),
        # Second linear layer
        nn.ReLU(),
        # 5 values for output layer, same length as RIGHT_ONLY action space
        nn.Linear(512, 5))

        if evaluation:
            for p in self.network.parameters():
                p.requires_grad = False

    def forward(self, x):
        return self.network(x)

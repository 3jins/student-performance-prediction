import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config

class NNModel(nn.Module):
    config = Config.instance()

    def __init__(self, is_training_mode):
        super(NNModel, self).__init__()
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.cuda else 'cpu')
        self.is_training_mode = is_training_mode
        # Names must be fc1, fc2. Otherwise, `torch.nn.parameters` cannot find them.
        self.fc1 = nn.Linear(self.config.INPUT_SIZE, self.config.HIDDEN_SIZES[0])
        self.fc2 = nn.Linear(self.config.HIDDEN_SIZES[0], self.config.OUTPUT_SIZE)

    def forward(self, inputs):
        linear = F.relu(self.fc1(inputs))
        linear = F.dropout(linear, training=self.is_training_mode)
        output = self.fc2(linear)
        return output

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config

class NNModel(nn.Module):
    config = Config.instance()
    training_mode = True

    def __init__(self, training_mode):
        super(NNModel, self).__init__()
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.cuda else 'cpu')
        self.training_mode = training_mode
        # Names must be fc1, fc2. Otherwise, `torch.nn.parameters` cannot find them.
        self.fc1 = nn.Linear(self.config.INPUT_SIZE, self.config.HIDDEN_SIZES[0])
        self.fc2 = nn.Linear(self.config.HIDDEN_SIZES[0], self.config.HIDDEN_SIZES[1])
        self.fc2_bn = nn.BatchNorm1d(self.config.HIDDEN_SIZES[1])
        self.fc3 = nn.Linear(self.config.HIDDEN_SIZES[1], self.config.HIDDEN_SIZES[2])
        self.fc3_bn = nn.BatchNorm1d(self.config.HIDDEN_SIZES[2])
        self.fc4 = nn.Linear(self.config.HIDDEN_SIZES[2], self.config.OUTPUT_SIZE)

    def forward(self, inputs):
        l1 = F.relu(self.fc1(inputs))
        l2 = F.relu(self.fc2_bn(self.fc2(l1)))
        l3 = F.relu(self.fc3_bn(self.fc3(l2)))
        l3 = F.dropout(l3, training=self.training_mode)
        output = self.fc4(l3)
        return output

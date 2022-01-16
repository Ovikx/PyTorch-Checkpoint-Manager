# Neural network source: https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html

import torch
import torch.nn as nn
from ckpt_manager import CheckpointManager
import time

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
print(net)

optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Create the CheckpointManager
manager = CheckpointManager(
    assets={
        'model' : net.state_dict(),
        'optimizer' : optimizer.state_dict()
    },
    directory='training_checkpoints',
    file_name='model',
    maximum=3,
    file_format='pt'
)

# Example of saving states
manager.save()

# Example of loading states
load_data = manager.load()
net.load_state_dict(load_data['model'])
optimizer.load_state_dict(load_data['optimizer'])
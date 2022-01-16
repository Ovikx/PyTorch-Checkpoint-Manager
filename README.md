# PyTorch Checkpoint Manager

A custom PyTorch checkpoint manager inspired by [TensorFlow's CheckpointManager](!https://www.tensorflow.org/api_docs/python/tf/train/CheckpointManager). Specify the necessary arguments in the constructor and then use the `CheckpointManager.save()` and `CheckpointManager.load()` methods to save/load models. Functionality is similar to that of `torch.save()` and `torch.load()`.

# Installation

Install via pip:
```cmd
pip install pytorch-ckpt-manager
```

# Example usage

The following is a simple convolutional network for demonstrating the checkpoint manager's functionality.

Imports:
```py
# Neural network source: https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html

import torch
import torch.nn as nn
from ckpt_manager import CheckpointManager
```

Create the neural network and its optimizer:
```py
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
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
```

Create the CheckpointManager:
```py
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
```

Save the states to the directory specified in the constructor:
```py
manager.save()
```

Load the states from the directory:
```py
load_data = manager.load()

net.load_state_dict(load_data['model'])
optimizer.load_state_dict(load_data['optimizer'])
```

If there is nothing to load, `net` and `optimizer` won't be altered.
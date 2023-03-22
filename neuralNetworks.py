import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import constants as c
import torch
torch.autograd.set_detect_anomaly(True)
class basic_DQN_Network(nn.Module):
    def __init__(self):
        super().__init__()

        # Determine a seed for the random weight generator, such that the generated weights are deterministic
        # i.e The generated 'random' weights are constantly the same
        # This is needed for reproducability of the results, cuz the initialized weights are important for convergence
        torch.manual_seed(c.DQN_SEED)

        self.fc1_dims = 256
        self.fc2_dims = 256
        self.fc3_dims = 256
        self.softmax = nn.Softmax(dim=-1)
        self.fc1 = nn.Linear(c.OBSERVATION_SPACE, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.fc3_dims)
        self.fc4 = nn.Linear(self.fc2_dims, c.ACTION_SPACE)

        self.optimizer = optim.Adam(self.parameters(), lr=c.LEARNING_RATE)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions = self.fc4(x)

        return actions

class hebbian_network_1_layer(nn.Module):
    def __init__(self):
        super().__init__()
        #torch.manual_seed(c.HEBBIAN_SEED)
        self.hebb_layer = nn.Linear(c.OBSERVATION_SPACE, c.ACTION_SPACE)

    def forward(self, state):
        x = self.hebb_layer(state)
        return x

class hebbian_network_2_layer(nn.Module):
    def __init__(self):
        super().__init__()
        #torch.manual_seed(c.HEBBIAN_SEED)

        self.layer_2_size = 32
        self.layer_1 = nn.Linear(c.OBSERVATION_SPACE, self.layer_2_size)
        self.layer_2 = nn.Linear(self.layer_2_size, c.ACTION_SPACE)

    def forward(self, state):
        x = self.layer_1(state)
        out = self.layer_2(x)
        return out


class hebbian_network_3_layer(nn.Module):
    def __init__(self):
        super().__init__()
        #torch.manual_seed(c.HEBBIAN_SEED)

        self.layer_2_size = 32
        self.layer_3_size = 32
        self.layer_1 = nn.Linear(c.OBSERVATION_SPACE, self.layer_2_size)
        self.layer_2 = nn.Linear(self.layer_2_size, self.layer_3_size)
        self.layer_3 = nn.Linear(self.layer_3_size, c.ACTION_SPACE)

    def forward(self, state):

        x = self.layer_1(state)
        x2 = self.layer_2(x)
        out = self.layer_3(x2)

        return out
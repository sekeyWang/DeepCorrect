import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Train on " + torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("Train on cpu")


def collate_fn(batch):
    batch_size = len(batch)
    input, target = [],[]
    maxLen = 0
    for i in range(batch_size):
        input.append(batch[i][0])
        target.append(batch[i][1])
        maxLen = max(maxLen, len(batch[i][1]))
    for i in range(batch_size):
        length = len(target[i])
        input[i] = np.concatenate((input[i], torch.zeros((1,67,maxLen-length))), 2)
        target[i] = np.concatenate((target[i], torch.zeros(maxLen-length)))

    input = np.array(input)
    target = np.array(target)

    input = torch.FloatTensor(input).to(device)
    target = torch.FloatTensor(target).to(device)

    return input, target


class Net(torch.nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 60, kernel_size=(86,5), padding=(0, 2))
        self.conv2 = nn.Conv1d(60, 60, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(60, 60, kernel_size=5, padding=2)
        self.conv4 = nn.Conv1d(60, 60, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(60, 1)
        self.double().to(device)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.view(1, 60, -1)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(60, -1).T
        x = self.fc1(x)
        x = torch.sigmoid(x)
        return x.view(1, -1)
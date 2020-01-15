import torch
import torch.nn as nn
import torch.nn.functional as F

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Train on " + torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("Train on cpu")


class Net(torch.nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 60, kernel_size=(64,5), padding=(0, 2))
        self.conv2 = nn.Conv1d(60, 60, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(60, 60, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(60, 1)
        self.double().to(device)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.view(1, 60, -1)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(60, -1).T
        x = self.fc1(x)
        x = torch.sigmoid(x)
        return x.view(1, -1)
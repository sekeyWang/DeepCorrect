import torch
import torch.optim as optim
import torch.nn as nn
import time
from torch.utils.data import random_split
from sklearn.model_selection import KFold

from model import Net, device
import config

criterion = nn.MSELoss()


def build_model(load_net_name=False):
    net = Net()
    if load_net_name is not False:
        net.load_state_dict(torch.load(load_net_name), map_location=device)
    return net


def cross_validation(dataset):
    cv = KFold(n_splits=config.splits, shuffle=True)
    for Fold, (train_index, val_index) in enumerate(cv.split(dataset)):
        print(f'Fold: {Fold + 1}')
        train(dataset, train_index, val_index)


def train(dataset, model) -> Net:
    optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, shuffle=False)
#    print(dataset[3474])
    for epoch in range(config.train_epochs):
        start = time.time()
        running_loss = 0
        for i, data in enumerate(dataloader, 0):
            if (i % 1000 == 0): print(i, len(dataloader))
            input, target = data
            model.zero_grad()
            output = model(input)
            loss = criterion(output, target)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
        print('Epoch %d/%d loss: %.3f time: %.3f' %(epoch + 1, config.train_epochs, running_loss / len(dataloader), time.time() - start))
        scheduler.step()
    return model


def test(dataset, model, index):
    dataloader = torch.utils.data.DataLoader(dataset=dataset[index], batch_size=1, shuffle=False)
    for i, data in enumerate(dataloader, 0):
        input, target = data
        output = model(input)
        test_result(output, target)


def test_result(output, target):
    pass

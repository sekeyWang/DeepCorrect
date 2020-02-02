import torch
import torch.optim as optim
import torch.nn as nn
import time
from torch.utils.data import random_split
from sklearn.model_selection import KFold

import config
from model import Net, device

import logging
logger = logging.getLogger(__name__)

criterion = nn.MSELoss()


def build_model(load_net_name=False):
    net = Net()
    if load_net_name is not False:
        net.load_state_dict(torch.load(load_net_name, map_location=device))
    return net

def save_model(net:Net, PATH):
    torch.save(net.state_dict(), PATH)

def cross_validation(dataset, model=build_model()):
    cv = KFold(n_splits=config.splits, shuffle=True)
    for Fold, (train_index, val_index) in enumerate(cv.split(dataset)):
        logger.info(f'Fold: {Fold + 1}')
        train(dataset[train_index], dataset[val_index], model=model)


def train(train_dataset, val_dataset, model) -> Net:
    optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=1, shuffle=False)
    for epoch in range(config.train_epochs):
        start = time.time()
        running_loss = 0
        for i, data in enumerate(dataloader, 0):
            if (i % 1000 == 0):
                logger.info('Trained:%d, Total: %d'%(i, len(dataloader)))
            input, target = data
            model.zero_grad()
            output = model(input)
            loss = criterion(output, target)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
        logger.info('Epoch %d/%d loss: %.3f time: %.3f' %(epoch + 1, config.train_epochs, running_loss / len(dataloader), time.time() - start))
        scheduler.step()
        test(val_dataset, model)
        save_model(model, "model/model1")
    return model


def test(dataset, model):
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    total_TP, total_FP, total_FN, total_TN = 0, 0, 0, 0
    for i, data in enumerate(dataloader, 0):
        input, target = data
        output = model(input)
        TP, FP, TN, FN = test_result(output[0], target[0], 0.50)
        total_TP, total_FP, total_FN, total_TN = total_TP + TP, total_FP + FP, total_FN + FN, total_TN + TN
    logger.info('TP=%d, FP=%d, FN=%d, TN=%d'%(total_TP, total_FP, total_FN, total_TN))
    logger.info('Precision = %.3f(%d/%d)' % (total_TP / (total_TP + total_FP), total_TP, (total_TP + total_FP)))
    logger.info('Recall = %.3f(%d/%d)' % (total_TP / (total_TP + total_FN), total_TP, (total_TP + total_FN)))

    total = total_TP + total_FP + total_FN + total_TN
    logger.info('Accuracy = %.3f(%d/%d)' % ((total_TP+total_TN) / total, total_TP+total_TN, total))

def test_result(output, target, threshold):
    TP, FP, TN, FN = 0, 0, 0, 0
    for idx in range(len(target)):
        if (output[idx] >= threshold):
            p = 1
        else:
            p = 0

        if target[idx] == 1:
            if p == 1:
                TP += 1
            else:
                FP += 1
        else:
            if p == 1:
                FN += 1
            else:
                TN += 1
    return TP, FP, TN, FN

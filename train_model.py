import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import time
from torch.utils.data import random_split
from sklearn.model_selection import KFold
import config
from model import Net, RNN_Net, device
from analysing_scripts import draw_single, draw_double, draw_triple
import numpy as np
from scipy.stats import pearsonr
from data_reader import train_Collate, test_Collate

import logging

logger = logging.getLogger(__name__)


class myMSEloss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.mean(torch.pow((x - y), 2))


criterion = nn.MSELoss()
criterion = nn.CrossEntropyLoss(ignore_index=2)  # input (N,C) terget (N)


def build_model(load_net_name=False):
    net = RNN_Net()
    if load_net_name is not False:
        net.load_state_dict(torch.load(load_net_name, map_location=device))
    return net


def save_model(net: Net, PATH):
    torch.save(net.state_dict(), PATH)


def cross_validation(dataset, model=build_model()):
    cv = KFold(n_splits=config.splits, shuffle=True)
    for Fold, (train_index, val_index) in enumerate(cv.split(dataset)):
        logger.info(f'Fold: {Fold + 1}')
        train(dataset[train_index], dataset[val_index], model=model)


def train(train_dataset, val_dataset, model) -> Net:
    # optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    # dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)
    dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=1, shuffle=True,
                                             collate_fn=train_Collate(), num_workers=10)
    for epoch in range(config.train_epochs):
        start = time.time()
        running_loss = 0
        cnt = 0
        correct_AA, total_AA = 0, 0
        for i, data in enumerate(dataloader, 0):
            if (i % 5000 == 4999):
                logger.info('Trained:%d, Total: %d' % (i, len(dataloader)))
                logger.info('Epoch %d/%d loss: %.3f time: %.3f' % (
                    epoch + 1, config.train_epochs, running_loss / cnt, time.time() - start))
                running_loss = 0
                cnt = 0
            # input, target = data
            input, input_lengths, target, target_lengths = data
            #            if (sum(target[0]) > len(target[0]) - 4): continue
            cnt += 1
            correct_AA += sum(target[0])
            total_AA += len(target[0])
            for j in range(len(target[0])):
                if (target[0][j] == 0): target[0][j] = -0.8
            input, target = input.to(device), target.to(device)
            input_lengths = input_lengths.to(device)
            target_lengths = target_lengths.to(device)
            model.zero_grad()
            output = model(input, input_lengths.to(device))
            new_target = target.view(-1).long()  # (batch*length)
            # loss = criterion(output, target)
            loss = criterion(output, new_target)
            running_loss += loss.item()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),max_norm=1)
            optimizer.step()
        print(loss.item())
        logger.info('Epoch %d/%d Train data accuracy: %.2f(%d/%d)' % (
            epoch + 1, config.train_epochs, correct_AA / total_AA, correct_AA, total_AA))
        # scheduler.step()
        output_list, target_list = test(val_dataset, model)
        analyze_result(output_list, target_list)
        #        analyze_distribution(output_list, target_list)
        save_model(model, "model/test3-7")
    return model


def get_list_score(input_feature, model: Net):
    input_tensor = torch.tensor([input_feature])
    output_tensor = model(input_tensor)
    output = output_tensor.detach().numpy()[0]
    return output


def test(dataset, model: Net):
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, shuffle=False, collate_fn=train_Collate(),
                                             num_workers=10)
    output_list, target_list = [], []
    for i, data in enumerate(dataloader, 0):
        # input, target = data
        input, input_lengths, target, target_lengths = data  # (batch,1,width,length), (batch,length)
        input, target = input.to(device), target.to(device)
        input_lengths, target_lengths = input_lengths.to(device), target_lengths.to(device)
        output = model(input, input_lengths.to(device)) # (batch,length,3)
        new_target = target.view(-1).long()
        #        if (sum(target[0]) > len(target[0])-6): continue
        output_list.append(output)
        # target_list.append(target)
        target_list.append(new_target)
    #        if (i > 200): break
    return output_list, target_list


def analyze_result(output_list, target_list):
    total_TP, total_FP, total_FN, total_TN = 0, 0, 0, 0
    for idx in range(len(output_list)):
        output = output_list[idx]  # (L,2)
        target = target_list[idx]  # (L,)

        output = torch.argmax(output, dim=1)  # (L)
        TP, FP, TN, FN = test_result(output, target)
        total_TP, total_FP, total_TN, total_FN = total_TP + TP, total_FP + FP, total_TN + TN, total_FN + FN
    logger.info('TP=%d, FP=%d, FN=%d, TN=%d' % (total_TP, total_FP, total_FN, total_TN))
    #    logger.info('Precision = %.3f(%d/%d)' % (total_TP / (total_TP + total_FP), total_TP, (total_TP + total_FP)))
    #    logger.info('Recall = %.3f(%d/%d)' % (total_TP / (total_TP + total_FN), total_TP, (total_TP + total_FN)))

    total = total_TP + total_FP + total_FN + total_TN
    logger.info('Accuracy = %.3f(%d/%d)' % ((total_TP + total_TN) / total, total_TP + total_TN, total))
    logger.info('Positive Accuracy: %.3f(%d/%d)' % (total_TP / (total_TP + total_FN), total_TP, (total_TP + total_FN)))
    logger.info('Negative Accuracy: %.3f(%d/%d)' % (total_TN / (total_TN + total_FP), total_TN, (total_TN + total_FP)))


def test_result(output, target):
    TP, FP, TN, FN = 0, 0, 0, 0
    for idx in range(len(target)):
        p = output[idx]
        if target[idx] == 1:
            if p == 1:
                TP += 1
            else:
                FN += 1
        else:
            if p == 1:
                FP += 1
            else:
                TN += 1
    return TP, FP, TN, FN


# def analyze_result(output_list, target_list, threshold=50):
#     total_TP, total_FP, total_FN, total_TN = 0, 0, 0, 0
#     for idx in range(len(output_list)):
#         output = output_list[idx]
#         target = target_list[idx]
#         TP, FP, TN, FN = test_result(output[0], target[0], threshold/100)
#         total_TP, total_FP, total_TN, total_FN = total_TP + TP, total_FP + FP, total_TN + TN, total_FN + FN
#     logger.info('TP=%d, FP=%d, FN=%d, TN=%d'%(total_TP, total_FP, total_FN, total_TN))
# #    logger.info('Precision = %.3f(%d/%d)' % (total_TP / (total_TP + total_FP), total_TP, (total_TP + total_FP)))
# #    logger.info('Recall = %.3f(%d/%d)' % (total_TP / (total_TP + total_FN), total_TP, (total_TP + total_FN)))
#
#     total = total_TP + total_FP + total_FN + total_TN
#     logger.info('Accuracy = %.3f(%d/%d)' % ((total_TP+total_TN) / total, total_TP+total_TN, total))
#     logger.info('Positive Accuracy: %.3f(%d/%d)' % (total_TP / (total_TP + total_FN), total_TP, (total_TP + total_FN)))
#     logger.info('Negative Accuracy: %.3f(%d/%d)' % (total_TN / (total_TN + total_FP+1), total_TN, (total_TN + total_FP+1)))
#
# def test_result(output, target, threshold):
#     TP, FP, TN, FN = 0, 0, 0, 0
#     for idx in range(len(target)):
#         if (output[idx] >= threshold):
#             p = 1
#         else:
#             p = 0
#
#         if target[idx] == 1:
#             if p == 1:
#                 TP += 1
#             else:
#                 FN += 1
#         else:
#             if p == 1:
#                 FP += 1
#             else:
#                 TN += 1
#     return TP, FP, TN, FN

def analyze_threshold(output_list, target_list):
    y1, y2, y3 = [], [], []
    x = range(1, 99)
    for threshold in x:
        total_TP, total_FP, total_FN, total_TN = 0, 0, 0, 0
        for idx in range(len(output_list)):
            output = output_list[idx]
            target = target_list[idx]
            TP, FP, TN, FN = test_result(output[0], target[0], threshold / 100)
            total_TP, total_FP, total_TN, total_FN = total_TP + TP, total_FP + FP, total_TN + TN, total_FN + FN
        y1.append(total_TN / (total_TN + total_FP + 1))
        y2.append(total_TP / (total_TP + total_FN + 1))
        y3.append((total_TP + total_TN) / (total_TP + total_FP + total_FN + total_TN))
        print(threshold, total_TN / (total_TN + total_FP + 1), total_TP / (total_TP + total_FN + 1))
    draw_triple(x, y1, y2, y3, 'title', 'Threshold', 'Accuracy', '0-accuracy', '1-accuracy', 'Accuracy')


def analyze_distribution(output_list, target_list):
    true_score_cnt, false_score_cnt = [0] * 100, [0] * 100
    true_score, false_score = [], []
    for idx in range(len(output_list)):
        output = output_list[idx][0]
        target = target_list[idx][0]
        for j in range(len(output)):
            if target[j] == 0:
                false_score.append(float(output[j]))
                false_score_cnt[int(output[j] * 100)] += 1
            else:
                true_score.append(float(output[j]))
                true_score_cnt[int(output[j] * 100)] += 1
    #    for i in range(1, 100):
    #        false_score[i] += false_score[i - 1]
    #        true_score[i] += true_score[i - 1]
    draw_single(range(100), false_score_cnt, 'title', 'score', 'count', 'false')
    draw_single(range(100), true_score_cnt, 'title', 'score', 'count', 'true')
    sumf = sum(false_score_cnt)
    sumt = sum(true_score_cnt)
    for i in range(100):
        false_score_cnt[i] /= sumf
        true_score_cnt[i] /= sumt
    draw_double(range(100), false_score_cnt, true_score_cnt, 'title', 'score', 'percent', 'false', 'true')
    lf, lt = len(false_score), len(true_score)
    false_score = sorted(false_score, reverse=True)
    true_score = sorted(true_score)
    logger.info("Mid: %.2f, P80: %.2f, P90: %.2f, P95: %.2f, P99: %.2f" % (
        false_score[lf // 2], false_score[lf // 5], false_score[lf // 10], false_score[lf // 20],
        false_score[lf // 100]))
    logger.info("Mid: %.2f, P80: %.2f, P90: %.2f, P95: %.2f, P99: %.2f" % (
        true_score[lt // 2], true_score[lt // 5], true_score[lt // 10], true_score[lt // 20], true_score[lt // 100]))


#    print(true_score_cnt, false_score_cnt)

def analyse_corr(dataset):
    for feature_num in range(0, 69):
        features = np.array([])
        results = np.array([])
        for idx, item in enumerate(dataset):
            feature, result = item
            features = np.append(features, feature[0][feature_num])
            results = np.append(results, result)
        #            if idx % 1000 == 999:
        #                print(idx)
        #        print(features.shape, results.shape)
        print(feature_num, pearsonr(features, results))

# Copyright (c) 201, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import re
import argparse
import os
import shutil
import time
import math
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torchvision.datasets
import torchvision.transforms as transforms
from mean_teacher import architectures, datasets, data, losses, ramps, cli
from mean_teacher.run_context import RunContext
from mean_teacher.data import NO_LABEL
from mean_teacher.utils import *
import sys
import codecs
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, Callback, CSVLogger, TensorBoard
from sklearn.metrics import average_precision_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
import keras
from util import *
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


LOG = logging.getLogger('main')

args = None
global_step = 0


def main(context):
    global global_step

    checkpoint_path = context.transient_dir
    training_log = context.create_train_log("training")

    dataset_config = datasets.__dict__[args.dataset]()
    num_classes = dataset_config.pop('num_classes')
    train_loader = create_data_loaders(**dataset_config, args=args)

    def create_model(ema=False):
        LOG.info("=> creating {pretrained}{ema}model '{arch}'".format(
            pretrained='pre-trained ' if args.pretrained else '',
            ema='EMA ' if ema else '',
            arch=args.arch))

        model_factory = architectures.__dict__[args.arch]
        model_params = dict(pretrained=args.pretrained,
                            num_classes=num_classes)
        model = model_factory(**model_params)
        model = nn.DataParallel(model).cuda()

        if ema:
            for param in model.parameters():
                param.detach_()

        return model

    model = create_model()

    # optimizer = torch.optim.SGD(model.parameters(), args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay,
    #                             nesterov=args.nesterov)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.000005)
    # scheduler = ReduceLROnPlateau(optimizer, 'min', patience = 5, min_lr = 0.5e-6, factor = np.sqrt(0.1))

    cudnn.benchmark = True

    for epoch in range(args.start_epoch, args.epochs):
        start_time = time.time()
        preci1 = train(train_loader, model, optimizer, epoch,
                       training_log)  # ema_model , scheduler
        # scheduler.step(preci1)
        LOG.info("--- training epoch in %s seconds ---" %
                 (time.time() - start_time))

        if args.checkpoint_epochs and (epoch + 1) % args.checkpoint_epochs == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'global_step': global_step,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, checkpoint_path, epoch + 1)

    start_time = time.time()
    LOG.info("Evaluating the primary model:")
    MA, MAP = evaluate_mAP(model)
    LOG.info("--- validation in %s seconds ---" %
             (time.time() - start_time))


def parse_dict_args(**kwargs):
    global args

    def to_cmdline_kwarg(key, value):
        if len(key) == 1:
            key = "-{}".format(key)
        else:
            key = "--{}".format(re.sub(r"_", "-", key))
        value = str(value)
        return key, value

    kwargs_pairs = (to_cmdline_kwarg(key, value)
                    for key, value in kwargs.items())
    cmdline_args = list(sum(kwargs_pairs, ()))
    args = parser.parse_args(cmdline_args)


def softmax(x):
    # print(np.shape(x))
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def create_data_loaders(train_transformation,
                        eval_transformation,
                        datadir,
                        args):
    traindir = os.path.join(datadir, args.train_subdir)

    dataset = torchvision.datasets.ImageFolder(traindir, train_transformation)

    if args.labels:
        with open(args.labels) as f:
            labels = dict(line.split(' ') for line in f.read().splitlines())
        labeled_idxs, _ = data.relabel_dataset(dataset, labels)

    sampler = SubsetRandomSampler(labeled_idxs)
    batch_sampler = BatchSampler(sampler, args.batch_size, drop_last=True)

    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_sampler=batch_sampler,
                                               num_workers=args.workers,
                                               pin_memory=True)

    return train_loader


def get_classes():
    """ 
        Read the action classes names and return them
        """
    actions = []
    with open('actions.txt', 'r+') as fid:
        # dummy 1st line
        l = fid.readline()
        for line in fid:
            c = line.split('\t')
            actions.append(c[0])
    return actions


def evaluate_mAP(model):
    classes = get_classes()  # classes names
    num_class = len(classes)  # 40
    base_dir = './imgs'
    test_list_file = open(base_dir + '/test.txt', 'r')
    test_image = []
    test_label = []

    for line in test_list_file:
        line = line.strip('\n').split(' ')
        test_image.append(line[0])
        test_label.append(int(line[1]))
    test_list_file.close()
    print(len(test_image), 'testing images ...')

    predictions = []  # np.zeros((num_class,len(test_image)))
    correct_cnt = 0
    image_size = (224, 224)
    for i in range(len(test_image)):
        img = load_img(base_dir + '/JPEGImages/' +
                       test_image[i], target_size=image_size)

        label = test_label[i]
        x = img_to_array(img)  # this is a Numpy array with shape (w,h,3)
        # x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, w,h,3)
        x = np.expand_dims(x, axis=0)
        x[:, :, :, 0] -= 103.939  # normalization
        x[:, :, :, 1] -= 116.77
        x[:, :, :, 2] -= 123.68

        x = x / 255.0
        x = torch.from_numpy(x)
        # print(np.shape(x))
        x = x.permute(0, 3, 1, 2)
        # print(np.shape(x))

        x = torch.autograd.Variable(x)

        score = softmax(model(x).data.cpu().numpy())
        # print(np.shape(score))

        predictions.append(list(score))  # [:,i] = score
        cls = np.argmax(score)
        if cls == test_label[i]:
            correct_cnt = correct_cnt + 1
        if i % 200 == 0:
            print(i, test_image[i], correct_cnt*1.0/len(test_image))
        #print(cls, classes[cls])

    # calculate the mean accuracy and mean average precision
    mA = []
    # predicted label for all inputs
    preds = [np.argmax(pre) for pre in predictions]
    print('accuracy:')
    for cls in range(num_class):
        label = [1 if y == cls else 0 for y in test_label]
        amt = sum(label)  # 计算每类数量
        pr = [1 if pre == cls else 0 for pre in preds]  # 把pred预测是这类的标1
        tp = sum([1 if p == 1 and l == 1 else 0 for p, l in zip(pr, label)])
        acc = tp/(amt+0.0000001)
        print(classes[cls] + ': ' + str(acc))
        mA.append(acc)
    print('mA:', sum(mA)/len(mA))

    print('average precision (AP):')
    mAP = 0
    ap_list = []
    for cls in range(num_class):
        samples = [pred[0][cls] for pred in predictions]
        labels = [1 if y == cls else 0 for y in test_label]
        # print(len(samples),len(labels),type(samples))
        # print(samples[0],labels[0])
        ap = average_precision_score(labels, samples)  # 可用于二分类或multilabel指示器格式
        ap_list.append(ap)
        mAP += ap
        print(classes[cls] + ': ' + str(ap))

    print('mAP:', mAP/num_class)
    return sum(mA)/len(mA), mAP/num_class


def train(train_loader, model, optimizer, epoch, log):
    global global_step

    class_criterion = nn.CrossEntropyLoss(
        size_average=False, ignore_index=NO_LABEL).cuda()

    meters = AverageMeterSet()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        meters.update('data_time', time.time() - end)

        # adjust_learning_rate(optimizer, epoch, i, len(train_loader))

        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target.cuda(async=True))

        minibatch_size = len(target_var)

        model_out = model(input_var)
        logit1 = model_out

        class_logit = logit1

        class_loss = class_criterion(class_logit, target_var) / minibatch_size
        meters.update('class_loss', class_loss.data[0])

        loss = class_loss
        assert not (np.isnan(loss.data[0]) or loss.data[0]
                    > 1e5), 'Loss explosion: {}'.format(loss.data[0])
        meters.update('loss', loss.data[0])

        prec1, prec5 = accuracy(class_logit.data, target_var.data, topk=(1, 5))
        meters.update('top1', prec1[0], minibatch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_step += 1

        # measure elapsed time
        meters.update('batch_time', time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            LOG.info(
                'Epoch: [{0}][{1}/{2}]\t'
                'Time {meters[batch_time]:.3f}\t'
                'Data {meters[data_time]:.3f}\t'
                'Class {meters[class_loss]:.4f}\t'
                'Prec@1 {meters[top1]:.3f}\t'.format(
                    epoch, i, len(train_loader), meters=meters))
            log.record(epoch + i / len(train_loader), {
                'step': global_step,
                **meters.values(),
                **meters.averages(),
                **meters.sums()
            })


def save_checkpoint(state, dirpath, epoch):
    filename = 'checkpoint.{}.ckpt'.format(epoch)
    checkpoint_path = os.path.join(dirpath, filename)
    best_path = os.path.join(dirpath, 'best.ckpt')
    torch.save(state, checkpoint_path)
    LOG.info("--- checkpoint saved to %s ---" % checkpoint_path)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    labeled_minibatch_size = max(target.ne(NO_LABEL).sum(), 1e-8)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / labeled_minibatch_size))
    return res


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = cli.parse_commandline_args()
    main(RunContext(__file__, 0))

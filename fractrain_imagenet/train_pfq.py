from __future__ import print_function

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np

import os
import shutil
import argparse
import time
import logging

import models
from data import *


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith('__')
                     and callable(models.__dict__[name])
                     )


def parse_args():
    # hyper-parameters are from ResNet paper
    parser = argparse.ArgumentParser(
        description='PFQ on ImageNet')
    parser.add_argument('--dir', help='annotate the working directory')
    parser.add_argument('--cmd', choices=['train', 'test'], default='train')
    parser.add_argument('--arch', metavar='ARCH', default='resnet50',
                        choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: cifar10_resnet_38)')
    parser.add_argument('--dataset', '-d', type=str, default='imagenet',
                        choices=['cifar10', 'cifar100','imagenet'],
                        help='dataset choice')
    parser.add_argument('--datadir', default='/home/yf22/dataset', type=str,
                        help='path to dataset')
    parser.add_argument('--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 4 )')
    parser.add_argument('--epoch', default=90, type=int,
                        help='number of epochs (default: 90)')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='manual iter number (useful on restarts)')
    parser.add_argument('--batch_size', default=256, type=int,
                        help='mini-batch size (default: 128)')
    parser.add_argument('--lr_schedule', default='piecewise', type=str,
                        help='learning rate schedule')
    parser.add_argument('--lr', default=0.1, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--print_freq', default=10, type=int,
                        help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str,
                        help='path to  latest checkpoint (default: None)')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pretrained model')
    parser.add_argument('--step_ratio', default=0.1, type=float,
                        help='ratio for learning rate deduction')
    parser.add_argument('--warm_up', action='store_true',
                        help='for n = 18, the model needs to warm up for 400 '
                             'iterations')
    parser.add_argument('--save_folder', default='save_checkpoints',
                        type=str,
                        help='folder to save the checkpoints')
    parser.add_argument('--eval_every', default=390, type=int,
                        help='evaluate model every (default: 1000) iterations')
    parser.add_argument('--num_bits',default=0,type=int,
                        help='num bits for weight and activation')
    parser.add_argument('--num_grad_bits',default=0,type=int,
                        help='num bits for gradient')
    parser.add_argument('--schedule', default=None, type=int, nargs='*',
                        help='precision schedule')
    parser.add_argument('--num_bits_schedule',default=None,type=int,nargs='*',
                        help='schedule for weight/act precision')
    parser.add_argument('--num_grad_bits_schedule',default=None,type=int,nargs='*',
                        help='schedule for grad precision')
    parser.add_argument('--act_fw', default=0, type=int,
                        help='precision of activation during forward, -1 means dynamic, 0 means no quantize')
    parser.add_argument('--act_bw', default=0, type=int,
                        help='precision of activation during backward, -1 means dynamic, 0 means no quantize')
    parser.add_argument('--grad_act_error', default=0, type=int,
                        help='precision of activation gradient during error backward, -1 means dynamic, 0 means no quantize')
    parser.add_argument('--grad_act_gc', default=0, type=int,
                        help='precision of activation gradient during weight gradient computation, -1 means dynamic, 0 means no quantize')
    parser.add_argument('--weight_bits', default=0, type=int,
                        help='precision of weight')
    parser.add_argument('--momentum_act', default=0.9, type=float,
                        help='momentum for act min/max')

    parser.add_argument('--num_turning_point', type=int, default=3)
    parser.add_argument('--initial_threshold', type=float, default=0.05)
    parser.add_argument('--decay', type=float, default=0.3)
    args = parser.parse_args()
    return args

# indicator
class loss_diff_indicator():
    def __init__(self, threshold, decay, epoch_keep=5):
        self.threshold = threshold
        self.decay = decay
        self.epoch_keep = epoch_keep
        self.loss = []
        self.scale_loss = 1
        self.loss_diff = [1 for i in range(1, self.epoch_keep)]

    def reset(self):
        self.loss = []
        self.loss_diff = [1 for i in range(1, self.epoch_keep)]

    def adaptive_threshold(self, turning_point_count):
        decay_1 = self.decay
        decay_2 = self.decay
        if turning_point_count == 1:
            self.threshold *= decay_1
        if turning_point_count == 2:
            self.threshold *= decay_2
        print('threshold decay to {}'.format(self.threshold))

    def get_loss(self, current_epoch_loss):
        if len(self.loss) < self.epoch_keep:
            self.loss.append(current_epoch_loss)
        else:
            self.loss.pop(0)
            self.loss.append(current_epoch_loss)

    def cal_loss_diff(self):
        if len(self.loss) == self.epoch_keep:
            for i in range(len(self.loss)-1):
                loss_now = self.loss[-1]
                loss_pre = self.loss[i]
                self.loss_diff[i] = np.abs(loss_pre - loss_now) / self.scale_loss
            return True
        else:
            return False

    def turning_point_emerge(self):
        flag = self.cal_loss_diff()
        if flag == True:
            print(self.loss_diff)
            for i in range(len(self.loss_diff)):
                if self.loss_diff[i] > self.threshold:
                    return False
            return True
        else:
            return False

def main():
    args = parse_args()
    global save_path
    save_path = args.save_path = os.path.join(args.save_folder, args.arch)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    models.ACT_FW = args.act_fw
    models.ACT_BW = args.act_bw
    models.GRAD_ACT_ERROR = args.grad_act_error
    models.GRAD_ACT_GC = args.grad_act_gc
    models.WEIGHT_BITS = args.weight_bits
    models.MOMENTUM = args.momentum_act

    args.num_bits = args.num_bits if not (args.act_fw + args.act_bw + args.grad_act_error + args.grad_act_gc + args.weight_bits) else -1

    # config logging file
    args.logger_file = os.path.join(save_path, 'log_{}.txt'.format(args.cmd))
    if os.path.exists(args.logger_file):
        os.remove(args.logger_file)
    handlers = [logging.FileHandler(args.logger_file, mode='w'),
                logging.StreamHandler()]
    logging.basicConfig(level=logging.INFO,
                        datefmt='%m-%d-%y %H:%M',
                        format='%(asctime)s:%(message)s',
                        handlers=handlers)

    global history_score
    history_score = np.zeros((args.epoch, 3))

    # initialize indicator
    # initial_threshold=0.15
    global scale_loss
    scale_loss = 0
    global my_loss_diff_indicator
    my_loss_diff_indicator = loss_diff_indicator(threshold=args.initial_threshold,
                                                 decay=args.decay)

    global turning_point_count
    turning_point_count = 0

    if args.cmd == 'train':
        logging.info('start training {}'.format(args.arch))
        run_training(args)

    elif args.cmd == 'test':
        logging.info('start evaluating {} with checkpoints from {}'.format(
            args.arch, args.resume))
        test_model(args)


def run_training(args):
    # create model
    training_loss = 0
    training_acc = 0

    model = models.__dict__[args.arch](args.pretrained)
    model = torch.nn.DataParallel(model).cuda()

    best_prec1 = 0
    best_epoch = 0

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logging.info('=> loading checkpoint `{}`'.format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])

            logging.info('=> loaded checkpoint `{}` (epoch: {})'.format(
                args.resume, checkpoint['epoch']
            ))
        else:
            logging.info('=> no checkpoint found at `{}`'.format(args.resume))

    cudnn.benchmark = False

    train_loader = prepare_train_data(dataset=args.dataset,
                                      datadir=args.datadir+'/train',
                                      batch_size=args.batch_size,
                                      shuffle=True,
                                      num_workers=args.workers)
    test_loader = prepare_test_data(dataset=args.dataset,
                                    datadir=args.datadir+'/val',
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=args.workers)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optimizer = torch.optim.Adam(model.parameters(), args.lr,
    #                             weight_decay=args.weight_decay)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    cr = AverageMeter()

    end = time.time()

    global scale_loss
    global history_score
    global turning_point_count
    global my_loss_diff_indicator

    for _epoch in range(args.start_epoch, args.epoch):
        lr = adjust_learning_rate(args, optimizer, _epoch)
        # adjust_precision(args, _epoch)
        adaptive_adjust_precision(args, turning_point_count)

        print('Learning Rate:', lr)
        print('num bits:', args.num_bits, 'num grad bits:', args.num_grad_bits)

        for i, (input, target) in enumerate(train_loader):
            # measuring data loading time
            data_time.update(time.time() - end)

            model.train()

            fw_cost = args.num_bits*args.num_bits/32/32
            eb_cost = args.num_bits*args.num_grad_bits/32/32
            gc_cost = eb_cost
            cr.update((fw_cost+eb_cost+gc_cost)/3)

            target = target.squeeze().long().cuda()
            input_var = Variable(input).cuda()
            target_var = Variable(target).cuda()

            # compute output
            output = model(input_var, args.num_bits, args.num_grad_bits)
            loss = criterion(output, target_var)
            training_loss += loss.item()

            # measure accuracy and record loss
            prec1, = accuracy(output.data, target, topk=(1,))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            training_acc += prec1.item()

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # print log
            if i % args.print_freq == 0:
                logging.info("Iter: [{0}][{1}/{2}]\t"
                             "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                             "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                             "Loss {loss.val:.3f} ({loss.avg:.3f})\t"
                             "Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t".format(
                                _epoch,
                                i,
                                len(train_loader),
                                batch_time=batch_time,
                                data_time=data_time,
                                loss=losses,
                                top1=top1)
                )

        epoch = _epoch + 1
        epoch_loss = training_loss / len(train_loader)
        with torch.no_grad():
            prec1 = validate(args, test_loader, model, criterion, _epoch)
            # prec_full = validate_full_prec(args, test_loader, model, criterion, i)
        history_score[epoch-1][0] = epoch_loss
        history_score[epoch-1][1] = np.round(training_acc / len(train_loader), 2)
        history_score[epoch-1][2] = prec1
        training_loss = 0
        training_acc = 0

        np.savetxt(os.path.join(save_path, 'record.txt'), history_score, fmt = '%10.5f', delimiter=',')

        # apply indicator
        # if epoch == 1:
        #     logging.info('initial loss value: {}'.format(epoch_loss))
        #     my_loss_diff_indicator.scale_loss = epoch_loss
        if epoch <= 10:
            scale_loss += epoch_loss
            logging.info('scale_loss at epoch {}: {}'.format(epoch, scale_loss / epoch))
            my_loss_diff_indicator.scale_loss = scale_loss / epoch
        if turning_point_count < args.num_turning_point:
            my_loss_diff_indicator.get_loss(epoch_loss)
            flag = my_loss_diff_indicator.turning_point_emerge()
            if flag == True:
                turning_point_count += 1
                logging.info('find {}-th turning point at {}-th epoch'.format(turning_point_count, epoch))
                # print('find {}-th turning point at {}-th epoch'.format(turning_point_count, epoch))
                my_loss_diff_indicator.adaptive_threshold(turning_point_count=turning_point_count)
                my_loss_diff_indicator.reset()

        logging.info('Epoch [{}] num_bits = {} num_grad_bits = {}'.format(epoch, args.num_bits, args.num_grad_bits))


        is_best = prec1 > best_prec1
        if is_best:
            best_prec1 = max(prec1, best_prec1)
            best_epoch = epoch
        #best_full_prec = max(prec_full, best_full_prec)

        print("Current Best Prec@1: ", best_prec1)
        logging.info("Current Best Epoch: {}".format(best_epoch))
        #print("Current Best Full Prec@1: ", best_full_prec)

        checkpoint_path = os.path.join(args.save_path, 'checkpoint_{:05d}_{:.2f}.pth.tar'.format(_epoch, prec1))
        save_checkpoint({
            'epoch': _epoch,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        },
            is_best, filename=checkpoint_path)
        shutil.copyfile(checkpoint_path, os.path.join(args.save_path,
                                                      'checkpoint_latest'
                                                      '.pth.tar'))


def validate(args, test_loader, model, criterion, _epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluation mode
    model.eval()
    end = time.time()
    for i, (input, target) in enumerate(test_loader):
        target = target.squeeze().long().cuda()
        input_var = Variable(input, volatile=True).cuda()
        target_var = Variable(target, volatile=True).cuda()

        # compute output
        output = model(input_var, args.num_bits, args.num_grad_bits)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, = accuracy(output.data, target, topk=(1,))
        top1.update(prec1.item(), input.size(0))
        losses.update(loss.item(), input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if (i % args.print_freq == 0) or (i == len(test_loader) - 1):
            logging.info(
                'Test: [{}/{}]\t'
                'Time: {batch_time.val:.4f}({batch_time.avg:.4f})\t'
                'Loss: {loss.val:.3f}({loss.avg:.3f})\t'
                'Prec@1: {top1.val:.3f}({top1.avg:.3f})\t'.format(
                    i, len(test_loader), batch_time=batch_time,
                    loss=losses, top1=top1
                )
            )

    logging.info('Epoch {} * Prec@1 {top1.avg:.3f}'.format(_epoch, top1=top1))
    return top1.avg


def validate_full_prec(args, test_loader, model, criterion, _epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluation mode
    model.eval()
    end = time.time()
    for i, (input, target) in enumerate(test_loader):
        target = target.squeeze().long().cuda()
        input_var = Variable(input, volatile=True).cuda()
        target_var = Variable(target, volatile=True).cuda()

        # compute output
        output = model(input_var, 0, 0)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, = accuracy(output.data, target, topk=(1,))
        top1.update(prec1.item(), input.size(0))
        losses.update(loss.item(), input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()


    logging.info('Epoch {} * Full Prec@1 {top1.avg:.3f}'.format(_epoch, top1=top1))
    return top1.avg


def test_model(args):
    # create model
    model = models.__dict__[args.arch](args.pretrained)
    model = torch.nn.DataParallel(model).cuda()

    if args.resume:
        if os.path.isfile(args.resume):
            logging.info('=> loading checkpoint `{}`'.format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            logging.info('=> loaded checkpoint `{}` (epoch: {})'.format(
                args.resume, checkpoint['epoch']
            ))
        else:
            logging.info('=> no checkpoint found at `{}`'.format(args.resume))

    cudnn.benchmark = False
    test_loader = prepare_test_data(dataset=args.dataset,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=args.workers)
    criterion = nn.CrossEntropyLoss().cuda()

    # validate(args, test_loader, model, criterion)

    with torch.no_grad():
        prec1 = validate(args, test_loader, model, criterion, args.start_iter)
        # prec_full = validate_full_prec(args, test_loader, model, criterion, args.start_iter)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        save_path = os.path.dirname(filename)
        shutil.copyfile(filename, os.path.join(save_path,
                                               'model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


schedule_cnt = 0
def adjust_precision(args, _epoch):
    if args.schedule:
        global schedule_cnt

        assert len(args.num_bits_schedule) == len(args.schedule) + 1
        assert len(args.num_grad_bits_schedule) == len(args.schedule) + 1

        if schedule_cnt == 0:
            args.num_bits = args.num_bits_schedule[0]
            args.num_grad_bits = args.num_grad_bits_schedule[0]
            schedule_cnt += 1

        for step in args.schedule:
            if _epoch == step:
                args.num_bits = args.num_bits_schedule[schedule_cnt]
                args.num_grad_bits = args.num_grad_bits_schedule[schedule_cnt]
                schedule_cnt += 1

def adaptive_adjust_precision(args, turning_point_count):
    args.num_bits = args.num_bits_schedule[turning_point_count]
    args.num_grad_bits = args.num_grad_bits_schedule[turning_point_count]

def adjust_learning_rate(args, optimizer, _epoch):
    lr = args.lr * (0.1 ** (_epoch // 30))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()

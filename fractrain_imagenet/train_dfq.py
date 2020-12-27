from __future__ import print_function

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np

from functools import reduce

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
        description='DFQ on ImageNet')
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
    parser.add_argument('--target_ratio',default=4,type=float,
                        help='target compression ratio')
    parser.add_argument('--target_ratio_schedule',default=None,type=float,nargs='*',
                        help='schedule for target compression ratio')
    parser.add_argument('--momentum_act', default=0.9, type=float,
                        help='momentum for act min/max')
    parser.add_argument('--relax', default=0, type=float,
                        help='relax parameter for target ratio') 
    parser.add_argument('--beta', default=1e-3, type=float,
                        help='coefficient')
    parser.add_argument('--computation_cost', default=True, type=bool,
                        help='using computation cost as regularization term')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
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
    handlers = [logging.FileHandler(args.logger_file, mode='w'),
                logging.StreamHandler()]
    logging.basicConfig(level=logging.INFO,
                        datefmt='%m-%d-%y %H:%M',
                        format='%(asctime)s:%(message)s',
                        handlers=handlers)

    if args.cmd == 'train':
        logging.info('start training {}'.format(args.arch))
        run_training(args)

    elif args.cmd == 'test':
        logging.info('start evaluating {} with checkpoints from {}'.format(
            args.arch, args.resume))
        test_model(args)

bits = [3, 4, 6, 8]
grad_bits = [6, 8, 12, 16]

def run_training(args):

    cost_fw = []
    for bit in bits:
        if bit == 0:
            cost_fw.append(1)
        else:
            cost_fw.append(bit/32)
    cost_fw = np.array(cost_fw) * args.weight_bits/32

    cost_eb = []
    for bit in grad_bits:
        if bit == 0:
            cost_eb.append(1)
        else:
            cost_eb.append(bit/32)
    cost_eb = np.array(cost_eb) * args.weight_bits/32

    cost_gc = []
    for i in range(len(bits)):
        if bits[i] == 0:
            cost_gc.append(1)
        else:
            cost_gc.append(bits[i]*grad_bits[i]/32/32)
    cost_gc = np.array(cost_gc)

    model = models.__dict__[args.arch](args.pretrained, proj_dim=len(bits))
    model = torch.nn.DataParallel(model).cuda()

    best_prec1 = 0
    best_epoch = 0
    best_full_prec = 0

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

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    cp_record = AverageMeter()
    cp_record_fw = AverageMeter()
    cp_record_eb = AverageMeter()
    cp_record_gc = AverageMeter()
    
    network_depth = sum(model.module.num_layers)

    layerwise_decision_statistics = []
    
    for k in range(network_depth):
        layerwise_decision_statistics.append([])
        for j in range(len(cost_fw)):
            ratio = AverageMeter()
            layerwise_decision_statistics[k].append(ratio)

    end = time.time()

    for _epoch in range(args.start_epoch, args.epoch):
        lr = adjust_learning_rate(args, optimizer, _epoch)
        adjust_target_ratio(args, _epoch)

        print('Learning Rate:', lr)
        print('Target Ratio:', args.target_ratio)

        for i, (input, target) in enumerate(train_loader):
            # measuring data loading time            
            data_time.update(time.time() - end)

            model.train()

            target = target.squeeze().long().cuda()
            input_var = Variable(input).cuda()
            target_var = Variable(target).cuda()

            output, masks = model(input_var, bits, grad_bits)
            
            computation_cost_fw = 0
            computation_cost_eb = 0
            computation_cost_gc = 0
            computation_all = 0
            
            for layer in range(network_depth):
                
                full_layer = reduce((lambda x, y: x * y), masks[layer][0].shape)
                
                computation_all  += full_layer
                
                for k in range(len(cost_fw)):
                    
                    dynamic_choice = masks[layer][k].sum()
                    
                    ratio = dynamic_choice / full_layer

                    layerwise_decision_statistics[layer][k].update(ratio.data, 1)
                    
                    computation_cost_fw += masks[layer][k].sum() * cost_fw[k]
                    computation_cost_eb += masks[layer][k].sum() * cost_eb[k]
                    computation_cost_gc += masks[layer][k].sum() * cost_gc[k]
            
            computation_cost = computation_cost_fw + computation_cost_eb + computation_cost_gc

            cp_ratio_fw = (float(computation_cost_fw) / float(computation_all)) * 100
            cp_ratio_eb = (float(computation_cost_eb) / float(computation_all)) * 100
            cp_ratio_gc = (float(computation_cost_gc) / float(computation_all)) * 100

            cp_ratio = (float(computation_cost) / float(computation_all*3)) * 100
                
            computation_cost *= args.beta

            if cp_ratio < args.target_ratio - args.relax:
                reg = -1
            elif cp_ratio >= args.target_ratio + args.relax:
                reg = 1
            elif cp_ratio >=args.target_ratio:
                reg = 0.1
            else:
                reg = -0.1
            
            loss_cls = criterion(output, target_var)

            if args.computation_cost:
                loss = loss_cls + computation_cost * reg
            else:
                loss = loss_cls

            # measure accuracy and record loss
            prec1, = accuracy(output.data, target, topk=(1,))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            cp_record.update(cp_ratio,1)
            cp_record_fw.update(cp_ratio_fw,1)
            cp_record_eb.update(cp_ratio_eb,1)
            cp_record_gc.update(cp_ratio_gc,1)

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
                             "Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t"
                             "Computation_Percentage: {cp_record.val:.3f}({cp_record.avg:.3f})\t"
                             "Computation_Percentage_FW: {cp_record_fw.val:.3f}({cp_record_fw.avg:.3f})\t"
                             "Computation_Percentage_EB: {cp_record_eb.val:.3f}({cp_record_eb.avg:.3f})\t"
                             "Computation_Percentage_GC: {cp_record_gc.val:.3f}({cp_record_gc.avg:.3f})\t".format(
                                _epoch,
                                i,
                                len(train_loader),
                                batch_time=batch_time,
                                data_time=data_time,
                                loss=losses,
                                top1=top1,
                                cp_record=cp_record,
                                cp_record_fw=cp_record_fw,
                                cp_record_eb=cp_record_eb,
                                cp_record_gc=cp_record_gc)
                )

        with torch.no_grad():
            prec1 = validate(args, test_loader, model, criterion, _epoch)
            # prec_full = validate_full_prec(args, test_loader, model, criterion, i)

        is_best = prec1 > best_prec1
        if is_best:
            best_prec1 = prec1
            best_epoch = _epoch
        #best_full_prec = max(prec_full, best_full_prec)

        print("Current Best Prec@1: ", best_prec1, "Best Epoch:", best_epoch)
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

    cost_fw = []
    for bit in bits:
        if bit == 0:
            cost_fw.append(1)
        else:
            cost_fw.append(bit/32)
    cost_fw = np.array(cost_fw) * args.weight_bits/32

    cost_eb = []
    for bit in grad_bits:
        if bit == 0:
            cost_eb.append(1)
        else:
            cost_eb.append(bit/32)
    cost_eb = np.array(cost_eb) * args.weight_bits/32

    cost_gc = []
    for i in range(len(bits)):
        if bits[i] == 0:
            cost_gc.append(1)
        else:
            cost_gc.append(bits[i]*grad_bits[i]/32/32)
    cost_gc = np.array(cost_gc)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    cp_record = AverageMeter()
    cp_record_fw = AverageMeter()
    cp_record_eb = AverageMeter()
    cp_record_gc = AverageMeter()
    
    network_depth = sum(model.module.num_layers)

    layerwise_decision_statistics = []
    
    for k in range(network_depth):
        layerwise_decision_statistics.append([])
        for j in range(len(cost_fw)):
            ratio = AverageMeter()
            layerwise_decision_statistics[k].append(ratio)

    model.eval()
    end = time.time()
    for i, (input, target) in enumerate(test_loader):
        target = target.squeeze().long().cuda()
        input_var = Variable(input, volatile=True).cuda()
        target_var = Variable(target, volatile=True).cuda()

        output, masks = model(input_var, bits, grad_bits)
        
        computation_cost_fw = 0
        computation_cost_eb = 0
        computation_cost_gc = 0
        computation_all = 0
        
        for layer in range(network_depth):
            
            full_layer = reduce((lambda x, y: x * y), masks[layer][0].shape)
            
            computation_all  += full_layer
            
            for k in range(len(cost_fw)):
                
                dynamic_choice = masks[layer][k].sum()
                
                ratio = dynamic_choice / full_layer

                layerwise_decision_statistics[layer][k].update(ratio.data, 1)
                
                computation_cost_fw += masks[layer][k].sum() * cost_fw[k]
                computation_cost_eb += masks[layer][k].sum() * cost_eb[k]
                computation_cost_gc += masks[layer][k].sum() * cost_gc[k]
        
        computation_cost = computation_cost_fw + computation_cost_eb + computation_cost_gc

        cp_ratio_fw = (float(computation_cost_fw) / float(computation_all)) * 100
        cp_ratio_eb = (float(computation_cost_eb) / float(computation_all)) * 100
        cp_ratio_gc = (float(computation_cost_gc) / float(computation_all)) * 100

        cp_ratio = (float(computation_cost) / float(computation_all*3)) * 100

        loss = criterion(output, target_var)

            # measure accuracy and record loss
        prec1, = accuracy(output.data, target, topk=(1,))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        cp_record.update(cp_ratio,1)
        cp_record_fw.update(cp_ratio_fw,1)
        cp_record_eb.update(cp_ratio_eb,1)
        cp_record_gc.update(cp_ratio_gc,1)

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 or (i == (len(test_loader) - 1)):
            logging.info("Iter: [{0}/{1}]\t"
                         "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                         "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                         "Loss {loss.val:.3f} ({loss.avg:.3f})\t"
                         "Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t"
                         "Computation_Percentage: {cp_record.val:.3f}({cp_record.avg:.3f})\t"
                         "Computation_Percentage_FW: {cp_record_fw.val:.3f}({cp_record_fw.avg:.3f})\t"
                         "Computation_Percentage_EB: {cp_record_eb.val:.3f}({cp_record_eb.avg:.3f})\t"
                         "Computation_Percentage_GC: {cp_record_gc.val:.3f}({cp_record_gc.avg:.3f})\t".format(
                            i,
                            len(test_loader),
                            batch_time=batch_time,
                            data_time=data_time,
                            loss=losses,
                            top1=top1,
                            cp_record=cp_record,
                            cp_record_fw=cp_record_fw,
                            cp_record_eb=cp_record_eb,
                            cp_record_gc=cp_record_gc)
            )

    logging.info('Epoch {} * Prec@1 {top1.avg:.3f}'.format(_epoch, top1=top1))
    
    for layer in range(network_depth):
        print('layer{}_decision'.format(layer + 1))
        for g in range(len(cost_fw)):
            print('{}_ratio{}'.format(g,layerwise_decision_statistics[layer][g].avg))

    return top1.avg


def validate_full_prec(args, test_loader, model, criterion, _epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    bits_full = np.zeros(len(bits))
    grad_bits_full = np.zeros(len(grad_bits))
    # switch to evaluation mode
    model.eval()
    end = time.time()
    for i, (input, target) in enumerate(test_loader):
        target = target.squeeze().long().cuda()
        input_var = Variable(input, volatile=True).cuda()
        target_var = Variable(target, volatile=True).cuda()

        # compute output
        output, _ = model(input_var, bits_full, grad_bits_full)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, = accuracy(output.data, target, topk=(1,))
        top1.update(prec1.item(), input.size(0))
        losses.update(loss.item(), input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if args.gate_type == 'rnn':
            model.module.control.repackage_hidden()

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
def adjust_target_ratio(args, _epoch):
    if args.schedule:
        global schedule_cnt

        assert len(args.target_ratio_schedule) == len(args.schedule) + 1

        if schedule_cnt == 0:
            args.target_ratio = args.target_ratio_schedule[0]
            schedule_cnt += 1

        for step in args.schedule:
            if _epoch == step:
                args.target_ratio = args.target_ratio_schedule[schedule_cnt]
                schedule_cnt += 1


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

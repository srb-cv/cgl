# after the model is trained, you might use convert_model.py to remove the data parallel module to make the model as standalone weight.
#
# Bolei Zhou

import argparse
import os
import shutil
import time

import math
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F

from model import alexnet
from regularizer import block_norm
from regularizer import receptive_fields

from tensorboardX import SummaryWriter
writer = SummaryWriter()


import wideresnet
import pdb
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


parser = argparse.ArgumentParser(description='PyTorch Places365 Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=6, type=int, metavar='N',
                    help='number of data loading workers (default: 6)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--penalty', '--penalty-lambda-weight', default=0.0, type=float,
                    metavar='PR', help='lambda penalty on the block norm regularizer')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--activation-penalty', '--ap', default=1e-5, type=float,
                    metavar='A', help='penalty fdr activation Regularizer (default: 1e-5)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--save', default='', type=str, metavar='PATH',
                    help='path where model file is to be saved (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_false',
                    help='use pre-trained model')
parser.add_argument('--num_classes',default=365, type=int, help='num of class in the model')
parser.add_argument('--dataset',default='places365',help='which dataset to train')
parser.add_argument('--gpu', default=0, type=int,
                    help = 'GPU id to use')


best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()
    print(args)
    # create model
    print("=> creating model '{}'".format(args.arch))
    if args.arch.lower().startswith('wideresnet'):
        # a customized resnet model with last feature map size as 14x14 for better class activation mapping
        model  = wideresnet.resnet50(num_classes=args.num_classes)
    elif args.arch.lower().startswith('alexnet'):
        model  = alexnet.Alexnet_module(num_classes=args.num_classes)
        regularizer = block_norm.RegularizeConvNetwork()
    else:
        model = models.__dict__[args.arch](num_classes=args.num_classes)

    if args.arch.lower().startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()
    print(model)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            #print("=> loaded optimizer parameter", optimizer.param_groups )
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)



    if args.evaluate:
        validate(val_loader, model, criterion, regularizer, epoch=0)
        return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, regularizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, regularizer, epoch)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict()
        }, is_best, args.arch.lower())


def train(train_loader, model, criterion, optimizer, regularizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    reg_losses = AverageMeter()
    activation_norm = AverageMeter()


    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output, conv_features = model(input)
        criterion_loss = criterion(output, target)

        if args.penalty == 0:
            weight_reg = torch.tensor(0.0, requires_grad=True).cuda()
            # loss = criterion_loss + weight_reg
        else:
            # print("Applying Block Norm Regularization")
            # compute the conv regularizers
            regulrizer_init = torch.tensor(0.0, requires_grad=True).cuda()
            weight_reg = regulrizer_init + regularizer.regularize_conv_layers(model, args.penalty)
            weight_reg = weight_reg.cuda()

        if args.activation_penalty == 0:
            activation_reg = torch.tensor(0.0, requires_grad=True).cuda()

        else:
            ### Preparing for activation norms
            act_regulrizer_init = torch.tensor(0.0, requires_grad=True).cuda()
            receptive_field = receptive_fields.SoftReceptiveField()
            soft_receptive_fields = receptive_field.calculate_receptive_field_layer_no_batch_norm(conv_features[0])
            assert (soft_receptive_fields.size() == conv_features[0].size())

            groupwise_activation_norm = regularizer.regularize_activation_groups_within_layer_full(conv_features[0])
            activation_reg = act_regulrizer_init + args.activation_penalty * groupwise_activation_norm.sum()
            # print(activation_reg)
            # activation_reg = torch.tensor(0.0).cuda()

        loss = criterion_loss + weight_reg + activation_reg







        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))
        reg_losses.update(weight_reg.item(), input.size(0))
        activation_norm.update(activation_reg.item(), input.size(0))

        # Display on tensorboard
        writer.add_scalar('train/loss', loss.item(), i)
        writer.add_scalar('train/reg_term', weight_reg.item(), i)
        writer.add_scalar('train/prec1', prec1[0], i)
        writer.add_scalar('train/prec5', prec5[0], i)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Reg_term {weight_reg.val:.4f} ({weight_reg.avg:.4f})\t'
                  'Activation_reg {activation_reg.val:.4f} ({activation_reg.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses,  weight_reg=reg_losses, activation_reg=activation_norm,
                   top1=top1, top5=top5))




def validate(val_loader, model, criterion, regularizer, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    reg_losses = AverageMeter()


    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking= True)
            target = target.cuda(args.gpu, non_blocking= True)

            # compute output
            output,_ = model(input)
            criterion_loss = criterion(output, target)

            if args.penalty == 0:
                weight_reg = torch.tensor(0.0, requires_grad=True).cuda()
                loss = criterion_loss + weight_reg
            else:
                # compute the conv regularizers
                regulrizer_init = torch.tensor(0.0, requires_grad=True).cuda()
                weight_reg = regulrizer_init + regularizer.regularize_conv_layers(model, args.penalty, eval=True)
                loss = criterion_loss + weight_reg

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            reg_losses.update(weight_reg.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Reg_term {weight_reg.val:.4f} ({weight_reg.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses, weight_reg=reg_losses,
                       top1=top1, top5=top5))

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

        # Display on tensorboard
        writer.add_scalar('val/loss', losses.avg, epoch)
        writer.add_scalar('val/reg_term', reg_losses.avg, epoch)
        writer.add_scalar('val/prec1', top1.avg, epoch)
        writer.add_scalar('train/prec5', top5.avg, epoch)
    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    if args.save:
        if os.path.isdir(args.save):
            torch.save(state, os.path.join(args.save, filename+'_latest.pth.tar'))
        else:
            print("Invalid Save Directory,\n Saving model in the working Directory")
            torch.save(state, filename + '_latest.pth.tar')
    else:
        torch.save(state, filename+'_latest.pth.tar')
    if is_best:
        if args.save:
            if os.path.isdir(args.save):
                shutil.copyfile(os.path.join(args.save,filename + '_latest.pth.tar'),
                                os.path.join(args.save,filename + '_best.pth.tar'))
            else:
                print("Invalid Save Directory, \n Saving model in the current working directory")
                shutil.copyfile(filename + '_latest.pth.tar', filename + '_best.pth.tar')
        else:
            shutil.copyfile(filename + '_latest.pth.tar', filename + '_best.pth.tar')


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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
    main()
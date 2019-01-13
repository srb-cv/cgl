# after the model is trained, you might use convert_model.py to remove the data parallel module to make the model as standalone weight.
#
# Bolei Zhou

import argparse
import os
import time

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import numpy as np

from model import alexnet
from regularizer import receptive_fields

from tensorboardX import SummaryWriter
writer = SummaryWriter()

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
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_false',
                    help='use pre-trained model')
parser.add_argument('--num_classes',default=365, type=int, help='num of class in the model')
parser.add_argument('--dataset',default='places365',help='which dataset to train')
parser.add_argument('--gpu', default=0, type=int,
                    help = 'GPU id to use')


best_prec1 = 0
unit_x = np.array([29, 39, 39, 46, 201, 195, 39, 39, 44, 203, 134])
unit_y = np.array([39, 210, 6, 99, 256, 225, 96, 102, 166, 241, 34])


def main():
    global args, best_prec1
    args = parser.parse_args()
    print(args)
    # create model
    print("=> creating model '{}'".format(args.arch))
    if args.arch.lower().startswith('alexnet'):
        model  = alexnet.Alexnet_module(num_classes=args.num_classes)
    else:
        model = models.__dict__[args.arch](num_classes=args.num_classes)

    if args.arch.lower().startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()
    print(model)


    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
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

    inspect_act_norms(train_loader, model)

    return


def inspect_act_norms(train_loader, model):
    batch_time = AverageMeter()
    difference_scores = AverageMeter()

    scores = []
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(train_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking= True)
            # target = target.cuda(args.gpu, non_blocking= True)

            # compute output
            output, conv_features = model(input)

            # act_regulrizer_init = torch.tensor(0.0, requires_grad=True).cuda()
            receptive_field = receptive_fields.SoftReceptiveField()
            soft_receptive_fields = receptive_field.calculate_receptive_field_layer_no_batch_norm(conv_features[0])
            assert (soft_receptive_fields.size() == conv_features[0].size())

            difference_score = mixed_version_measure(unit_x, unit_y, soft_receptive_fields)
            difference_score = torch.stack(difference_score)

            difference_scores.update(difference_score, len(difference_score))
            scores.append(difference_score)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print(
                      'Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Score {difference_score.val} ({difference_score.avg})'.format(
                          i, len(train_loader), batch_time=batch_time, difference_score=difference_scores))

    print("Avg Score : {} and Total Score {}".format(sum(scores)/len(scores), sum(scores)))
    return None


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


def purely_diff_based_measure(unit_x, unit_y, receptive_fields):
    #print(receptive_fields)
    act_unit_x = receptive_fields[:, unit_x - 1]
    act_unit_y = receptive_fields[:, unit_y - 1]
    difference_tensor = act_unit_x - act_unit_y
    score = []
    for i in range(len(unit_x)):
        score.append((2 * difference_tensor[:, i].norm(1))
                     / (act_unit_x[:, i].norm(1) + act_unit_y[:, i].norm(1) + difference_tensor[:, i].norm(1)))
    return score


def dicer_version_measure(unit_x, unit_y, receptive_fields):
    act_unit_x = receptive_fields[:, unit_x - 1]
    act_unit_y = receptive_fields[:, unit_y - 1]
    hadamard_prod = torch.mul(act_unit_x, act_unit_y)
    score = []
    for i in range(len(unit_x)):
        iou_score = hadamard_prod[:, i].norm(1) / (act_unit_x[:, i].norm(1) + act_unit_y[:, i].norm(1) -
                                                   hadamard_prod[:, i].norm(1))
        score.append(1 - iou_score)
    return score


def mixed_version_measure(unit_x, unit_y, receptive_fields):
    act_unit_x = receptive_fields[:, unit_x - 1]
    act_unit_y = receptive_fields[:, unit_y - 1]
    hadamard_prod = torch.mul(act_unit_x, act_unit_y)
    difference_tensor = act_unit_x -act_unit_y
    score = []
    for i in range(len(unit_x)):
        iou_score = difference_tensor[:, i].norm(1) / (act_unit_x[:, i].norm(1) + act_unit_y[:, i].norm(1) -
                                                   hadamard_prod[:, i].norm(1))
        score.append(1 - iou_score)
    return score


if __name__ == '__main__':
    main()

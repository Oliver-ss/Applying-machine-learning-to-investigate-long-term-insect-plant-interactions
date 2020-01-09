from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from model import build_ssd
import os
import sys
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse
from cycle_learning_rate import cycle_lr
from common import config
import json

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()

parser.add_argument('--train_label', default='../../Data/Labels/label_train_new.json',
                    help='Train label path')
parser.add_argument('--val_label', default='../../Data/Labels/Validation-3class.json',
                    help='Val label path')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--visdom', default=False, type=str2bool,
                    help='Use visdom for loss visualization')
parser.add_argument('--save_folder', default='train_log/',
                    help='Directory for saving checkpoint models')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU device number')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)
if not os.path.exists(args.save_folder + 'eval/'):
    os.mkdir(args.save_folder + 'eval/')
if not os.path.exists(args.save_folder + 'model/'):
    os.mkdir(args.save_folder + 'model/')


if args.visdom:
    import visdom
    viz = visdom.Visdom()

def train():
    cfg = config.Damage
    train_dataset = Damage_Dataset(name='train', label_root=args.train_label,
                               transform=SSDAugmentation())
    val_dataset = Damage_Dataset(name='validation', label_root=args.val_label,
            transform=BaseTransform())


    ssd_net = build_ssd('train', cfg['min_dim'], config.num_classes)
    net = ssd_net
    
    #cycle_cos_lr = cycle_lr(500, cfg['peak_lr'], cfg['T_init'], cfg['T_warmup'])

    if args.cuda:
        net = torch.nn.DataParallel(ssd_net)
        cudnn.benchmark = True

    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        ssd_net.load_weights(args.resume)
    else:
        vgg_weights = torch.load('../../pretrained/vgg16_reducedfc.pth')
        print('Loading base network...')
        ssd_net.vgg.load_state_dict(vgg_weights)

    if args.cuda:
        net = net.cuda()

    if not args.resume:
        print('Initializing weights...')
        # initialize newly added layers' weights with xavier method
        ssd_net.extras.apply(weights_init)
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)

    optimizer = optim.SGD(net.parameters(), lr=config.lr, momentum=config.momentum,
                          weight_decay=config.weight_decay)
    criterion = MultiBoxLoss(config.num_classes, overlap_thresh=0.5,
                             prior_for_matching=True, bkg_label=0,
                             neg_mining=True, neg_pos=3, neg_overlap=0.5,
                             encode_target=False, use_gpu=args.cuda)

    net.train()
    # loss counters
    loc_loss = 0
    conf_loss = 0
    epoch = 0
    print('Loading the train dataset...')

    epoch_size = len(train_dataset) // config.batch_size
    print('Training SSD on:', train_dataset.name)
    print('Using the specified args:')
    print(args)

    step_index = 0

    if args.visdom:
        vis_title = os.getcwd().split('/')[-1]
        vis_legend = ['Loc Loss', 'Conf Loss', 'Total Loss']
        iter_plot = create_vis_plot('Iteration', 'Loss', vis_title, vis_legend)
        #epoch_plot = create_vis_plot('Epoch', 'Loss', vis_title, vis_legend)
        iter_val_plot = create_vis_plot('Iteration', 'Val Loss', vis_title, vis_legend)

    data_loader = data.DataLoader(train_dataset, config.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)

    val_data_loader = data.DataLoader(val_dataset, config.batch_size,
                                      num_workers=args.num_workers,shuffle=True,
                                      collate_fn=detection_collate, pin_memory=True)
    # create batch iterator
    batch_iterator = iter(data_loader)
    val_batch_iterator = iter(val_data_loader)
    for iteration in range(args.start_iter, config.max_iter):
        #if args.visdom and iteration != 0 and (iteration % epoch_size == 0):
        #    update_vis_plot(epoch, loc_loss, conf_loss, epoch_plot, None,
        #                    'append', epoch_size)
        #    # reset epoch loss counters
        #    loc_loss = 0
        #    conf_loss = 0
        #    epoch += 1

        if iteration in config.lr_steps:
            step_index += 1
            adjust_learning_rate(optimizer, config.gamma, step_index)

        # cycle lr
        #for param_group in optimizer.param_groups:
        #    param_group['lr'] = cycle_cos_lr.get_lr(iteration)

        # load train data
        try:
            images, targets = next(batch_iterator)
        except StopIteration:
            batch_iterator = iter(data_loader)
            images, targets = next(batch_iterator)

        if args.cuda:
            images = Variable(images.cuda())
            targets = [Variable(ann.cuda(), volatile=True) for ann in targets]
        else:
            images = Variable(images)
            targets = [Variable(ann, volatile=True) for ann in targets]

        # forward
        t0 = time.time()
        out = net(images)
        # backprop
        optimizer.zero_grad()
        loss_l, loss_c = criterion(out, targets)
        loss = loss_l + loss_c
        loss.backward()
        optimizer.step()
        t1 = time.time()
        loc_loss += loss_l.item()
        conf_loss += loss_c.item()


        if iteration % 10 == 0:
            print('timer: %.4f sec.' % (t1 - t0))
            print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.item()), end=' ')
            if args.visdom:
                viz.line(
                    X=torch.ones((1, 3)).cpu() * iteration,
                    Y=torch.Tensor([loss_l, loss_c, loss]).unsqueeze(0).cpu(),
                    win=iter_plot,
                    update='True' if iteration == 10 else 'append'
                    )

        if iteration % 100 == 0 and iteration != 0:
            val_loss_l, val_loss_c, val_loss = val(net, val_data_loader, criterion)
            print('Val_Loss: %.4f ||' % (val_loss.item()), end=' ')
            if args.visdom:
                viz.line(
                    X=torch.ones((1, 3)).cpu() * iteration,
                    Y=torch.Tensor([val_loss_l, val_loss_c, val_loss]).unsqueeze(0).cpu(),
                    win=iter_val_plot,
                    update='True' if iteration == 100 else 'append'
                )

        #if args.visdom:
            #update_vis_plot(iteration, loss_l.data[0], loss_c.data[0],
        #    update_vis_plot(iteration, loss_l.item(), loss_c.item(),
        #                    iter_plot, epoch_plot, 'append')

        if iteration != 0 and iteration % 1000 == 0:
            print('Saving state, iter:', iteration)
            torch.save(ssd_net.state_dict(), args.save_folder + 'model/'
                    + repr(iteration) + '.pth')
            loss_file = {'loss': val_loss.item(), 'loc_loss': val_loss_l.item(), 'conf_loss': val_loss_c.item()}
            with open(os.path.join(args.save_folder, 'eval', repr(iteration)+'.json'), 'w') as f:
                json.dump(loss_file, f)

    torch.save(ssd_net.state_dict(),
            args.save_folder + '' + 'leaves' + '.pth')

def val(model, dataloader, criterion):
    model.eval()  # evaluation mode

    loc_loss = 0
    conf_loss = 0
    num_img = 0

    for iteration, (images, targets) in enumerate(dataloader):
        num_img += 1
        if args.cuda:
            images = Variable(images.cuda())
            targets = [Variable(ann.cuda(), volatile=True) for ann in targets]
        else:
            images = Variable(images)
            targets = [Variable(ann, volatile=True) for ann in targets]
        out = model(images)
        with torch.no_grad():
            loss_l, loss_c = criterion(out, targets)
            loc_loss += loss_l.data
            conf_loss += loss_c.data

    loc_loss /= num_img
    conf_loss /= num_img
    loss = loc_loss + conf_loss
    model.train()

    return loc_loss, conf_loss, loss

def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = config.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def xavier(param):
    init.xavier_uniform(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


def create_vis_plot(_xlabel, _ylabel, _title, _legend):
    return viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, 3)).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )


def update_vis_plot(iteration, loc, conf, window1, window2, update_type,
                    epoch_size=1):
    viz.line(
        X=torch.ones((1, 3)).cpu() * iteration,
        Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu() / epoch_size,
        win=window1,
        update=update_type
    )
    # initialize epoch plot on first iteration
    if iteration == 0:
        viz.line(
            X=torch.zeros((1, 3)).cpu(),
            Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu(),
            win=window2,
            update=True
        )


if __name__ == '__main__':
    train()

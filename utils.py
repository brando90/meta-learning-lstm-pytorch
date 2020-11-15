from __future__ import division, print_function, absolute_import

import os
import pdb
import logging

import torch
import numpy as np


class GOATLogger:

    def __init__(self, args):
        args.save = f'{args.save}/log-{args.seed}'.format(args.seed)

        self.mode = args.mode
        self.save_root = args.save  # path to this specific log for this expt
        self.log_freq = args.log_freq

        if self.mode == 'train':
            if not os.path.exists(self.save_root):
                os.mkdir(self.save_root)
            filename = os.path.join(self.save_root, 'console.log')
            logging.basicConfig(level=logging.DEBUG,
                format='%(asctime)s.%(msecs)03d - %(message)s',
                datefmt='%b-%d %H:%M:%S',
                filename=filename,
                filemode='w')
            console = logging.StreamHandler()
            console.setLevel(logging.INFO)
            console.setFormatter(logging.Formatter('%(message)s'))
            logging.getLogger('').addHandler(console)

            logging.info("Logger created at {}".format(filename))
        else:
            logging.basicConfig(level=logging.INFO,
                format='%(asctime)s.%(msecs)03d - %(message)s',
                datefmt='%b-%d %H:%M:%S')

        logging.info("Random Seed: {}".format(args.seed))
        self.reset_stats()

    def reset_stats(self):
        if self.mode == 'train':
           self.stats = {'train': {'loss': [], 'acc': []},
                          'eval': {'loss': [], 'acc': []}}
        else:
            self.stats = {'eval': {'loss': [], 'acc': []}}

    def batch_info(self, **kwargs):
        if kwargs['phase'] == 'train':
            self.stats['train']['loss'].append(kwargs['loss'])
            self.stats['train']['acc'].append(kwargs['acc'])

            if kwargs['eps'] % self.log_freq == 0 and kwargs['eps'] != 0:
                loss_mean = np.mean(self.stats['train']['loss'])
                acc_mean = np.mean(self.stats['train']['acc'])
                #self.draw_stats()
                self.loginfo("[{:5d}/{:5d}] loss: {:6.4f} ({:6.4f}), acc: {:6.3f}% ({:6.3f}%)".format(\
                    kwargs['eps'], kwargs['totaleps'], kwargs['loss'], loss_mean, kwargs['acc'], acc_mean))

        elif kwargs['phase'] == 'eval':
            self.stats['eval']['loss'].append(kwargs['loss'])
            self.stats['eval']['acc'].append(kwargs['acc'])

        elif kwargs['phase'] == 'evaldone':
            loss_mean = np.mean(self.stats['eval']['loss'])
            loss_std = np.std(self.stats['eval']['loss'])
            acc_mean = np.mean(self.stats['eval']['acc'])
            acc_std = np.std(self.stats['eval']['acc'])
            self.loginfo("[{:5d}] Eval ({:3d} episode) - loss: {:6.4f} +- {:6.4f}, acc: {:6.3f} +- {:5.3f}%".format(\
                kwargs['eps'], kwargs['totaleps'], loss_mean, loss_std, acc_mean, acc_std))

            self.reset_stats()
            return acc_mean

        else:
            raise ValueError("phase {} not supported".format(kwargs['phase']))

    def logdebug(self, strout):
        logging.debug(strout)
    def loginfo(self, strout):
        logging.info(strout)


def accuracy(output, target, topk=(1,)):
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
        return res[0].item() if len(res) == 1 else [r.item() for r in res]


def save_ckpt(episode, metalearner, optim, save):
    if not os.path.exists(os.path.join(save, 'ckpts')):
        os.mkdir(os.path.join(save, 'ckpts'))

    torch.save({
        'episode': episode,
        'metalearner': metalearner.state_dict(),
        'optim': optim.state_dict()
    }, os.path.join(save, 'ckpts', 'meta-learner-{}.pth.tar'.format(episode)))


def resume_ckpt(metalearner, optim, resume, device):
    ckpt = torch.load(resume, map_location=device)
    last_episode = ckpt['episode']
    metalearner.load_state_dict(ckpt['metalearner'])
    optim.load_state_dict(ckpt['optim'])
    return last_episode, metalearner, optim


def preprocess_grad_loss(x, p=10, eps=1e-8):
    """ Preprocessing (vectorized) implementation from the paper:

    if |x| >= e^-p (not too small)
        coord1, coord2 = (log(|x| + eps)/p, sign(x))
    else: (too small
        coord1, coord2 = (-1, (e^p)*x)
    return stack(coord1,coord2)
    
    usually applied to loss and grads.

    Arguments:
        x {[torch.Tensor]} -- input to preprocess
    
    Keyword Arguments:
        p {int} -- number that indicates the scaling (default: {10})
        eps {float} - numerical stability param (default: {1e-8})
    
    Returns:
        [torch.Tensor] -- preprocessed numbers
    """
    # implements vectorized if statement
    indicator = (x.abs() >= np.exp(-p)).to(torch.float32)

    # preproc1 - magnitude path (coord 1) log(|x|)/(p+eps) or -1
    # if not too small use the exponent of the magnitude/p
    # if too small use a -1 to indicate too small to the neural net
    x_proc1 = indicator * torch.log(x.abs() + eps) / p + (1 - indicator) * -1
    # preproc2 - sign path (coord 2) sign(x) or (e^p)*x
    # if not too small log(|x|)/p
    # if too small (e^p)*x
    x_proc2 = indicator * torch.sign(x) + (1 - indicator) * np.exp(p) * x
    # stack
    # usually in meta-lstm x is n_learner_params so this forms a tensor of size [n_learnaer_params, 2]
    x_proc = torch.stack((x_proc1, x_proc2), 1)
    return x_proc


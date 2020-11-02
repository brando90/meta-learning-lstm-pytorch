from __future__ import division, print_function, absolute_import

import os
import re
import pdb
import glob
import pickle

import torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import PIL.Image as Image
import numpy as np

from tqdm import tqdm

from pathlib import Path

from types import SimpleNamespace

"""
The task sampling process here works like this.
- Sample N-classes (randomly). 
- Sample the dataloaders for each class using the random labels.
- Sample the 5+15=20 examples for each class and joing them to form the N-way, K-shot task with K_eval
    query examples.
- Form the task N*(K+K_eval)=5*20=100 dataset for meta-training/eval
"""

class AllMiniImagenetDataset(data.Dataset):

    def __init__(self, root, phase='train', n_shot=5, n_eval=15, transform=None):
        """loads data loaders for all image classes/labels and all the images """
        # path to split e.g. '/Users/brando/data/miniimagenet_meta_lstm/miniImagenet/train'
        root = os.path.join(root, phase)
        # list of strings with the label name
        self.labels = sorted(os.listdir(root))
        # list of path to images (str) per class
        images = [glob.glob(os.path.join(root, label, '*')) for label in self.labels]

        self.class_loaders = []
        # loops through each labels for the current split e.g. 64 for train
        for class_idx, _ in enumerate(self.labels):
            class_name = class_idx
            imgs = images[class_idx]  # 600 imagesi for label
            # create a dataset for the current label/class
            labeldataset = LabelDataset(images=imgs, label=class_name, transform=transform)  # all 600 images for a specific class=label
            # create a list of data loaders for each class/label
            class_loaders = data.DataLoader(labeldataset, batch_size=n_shot+n_eval, shuffle=True, num_workers=0)  # by sampling from 600 images we sample a task for the meta-learner to train
            self.class_loaders.append(class_loaders)
        print(f'len(self.class_loaders) = {len(self.class_loaders)}')

    def __getitem__(self, class_idx):
        """ return support+query set for a specific class/label"""
        classloader = iter(self.class_loaders[class_idx])
        # sample batch of size 5+15 = 20 from 600 available for current label
        support_query_set_current_label = next(classloader)
        return support_query_set_current_label

    def __len__(self):
        return len(self.class_loaders)  # e.g. 64

class LabelDataset(data.Dataset):
    """
    Dataset that holds all the images for a specific class. e.g. holds 600 images.
    """

    def __init__(self, images, label, transform=None):
        self.images = images
        self.label = label
        self.transform = transform

    def __getitem__(self, idx):
        """gets an image and it's label"""
        image = Image.open(self.images[idx]).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        return image, self.label

    def __len__(self):
        return len(self.images)


# class EpisodicSampler(data.Sampler):
class TaskSampler(data.Sampler):
    """
    Helps in the creation of a N-way, K-shot task by getting the random N classes/labels
    """

    def __init__(self, total_classes, n_class, n_episode):
        self.total_classes = total_classes  # e.g. 64
        self.n_class = n_class  # N
        self.n_episode = n_episode

    def __iter__(self):
        """ gets 5 random labels for a task """
        for i in range(self.n_episode):
            # permute the list of labels
            torch.randperm(self.total_classes)
            # get N of the labels
            random_N_labels = torch.randperm(self.total_classes)[:self.n_class]
            yield random_N_labels

    def __len__(self):
        return self.n_episode


def prepare_data(args):

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    train_set = AllMiniImagenetDataset(args.data_root, 'train', args.n_shot, args.n_eval,
        transform=transforms.Compose([
            transforms.RandomResizedCrop(args.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.2),
            transforms.ToTensor(),
            normalize]))
    print(f'size of train_set (meta) = {len(train_set)}')

    val_set = AllMiniImagenetDataset(args.data_root, 'val', args.n_shot, args.n_eval,
        transform=transforms.Compose([
            transforms.Resize(args.image_size * 8 // 7),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            normalize]))

    test_set = AllMiniImagenetDataset(args.data_root, 'test', args.n_shot, args.n_eval,
        transform=transforms.Compose([
            transforms.Resize(args.image_size * 8 // 7),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            normalize]))

    train_loader = data.DataLoader(train_set, num_workers=args.n_workers, pin_memory=args.pin_mem,
        batch_sampler=TaskSampler(len(train_set), args.n_class, args.episode))

    val_loader = data.DataLoader(val_set, num_workers=2, pin_memory=False,
        batch_sampler=TaskSampler(len(val_set), args.n_class, args.episode_val))

    test_loader = data.DataLoader(test_set, num_workers=2, pin_memory=False,
        batch_sampler=TaskSampler(len(test_set), args.n_class, args.episode_val))

    return train_loader, val_loader, test_loader

def brandos_load(args):
    args.mode = "train"
    args.n_shot = 5
    args.n_eval = 15
    args.n_class = 5
    args.input_size = 4
    args.hidden_size = 20
    args.lr = 1e-3
    args.episode = 50000
    args.episode_val = 100
    args.epoch = 8
    args.batch_size = 25  # N*K = 5*5
    args.image_size = 84
    args.grad_clip = 0.25
    args.bn_momentum = 0.95
    args.bn_eps = 1e-3
    args.data = "miniimagenet"
    args.data_root = Path('~/data/miniimagenet_meta_lstm/miniImagenet/').expanduser()
    args.pin_mem = True
    args.log_freq = 50
    args.val_freq = 10
    args.cpu = True
    args.n_workers = 4
    return args

def main():
    args = SimpleNamespace()
    args = brandos_load(args)
    train_loader, val_loader, test_loader = prepare_data(args)


if __name__ == '__main__':
    main()
    print('DONE')

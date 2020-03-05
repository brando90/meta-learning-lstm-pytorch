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
import PIL.Image as PILI
import numpy as np

from tqdm import tqdm


class EpisodeDataset(data.Dataset):

    def __init__(self, root, phase='train', n_shot=5, n_eval=15, transform=None):
        """Args:
            root (str): path to data
            phase (str): train, val or test
            n_shot (int): how many examples per class for training (k/n_support)
            n_eval (int): how many examples per class for evaluation
                - n_shot + n_eval = batch_size for data.DataLoader of ClassDataset
            transform (torchvision.transforms): data augmentation
        """
        root = os.path.join(root, phase)
        self.labels = sorted(os.listdir(root))
        images = [glob.glob(os.path.join(root, label, '*')) for label in self.labels]

        # self.episode_loader = [data.DataLoader(
        #     ClassDataset(images=images[idx], label=idx, transform=transform),
        #     batch_size=n_shot+n_eval, shuffle=True, num_workers=0) for idx, _ in enumerate(self.labels)]

        self.episode_loader = []
        # loops through each labels for the meta set split e.g. 64, 16, 20
        for idx, _ in enumerate(self.labels):
            #print(f'idx = {idx}')
            label = idx
            imgs = images[idx] # 600 images
            #print(f'# of images in Class Dataset = {len(imgs)}')
            classdataset = ClassDataset(images=imgs, label=label, transform=transform) # all 600 images for a specific class=label
            taskloader = data.DataLoader(classdataset, batch_size=n_shot+n_eval, shuffle=True, num_workers=0) # by sampling from 600 images we sample a data set split for the meta-learner to train
            self.episode_loader.append(taskloader)
        print(f'len(self.episode_loader) = {len(self.episode_loader)}')

    def __getitem__(self, idx):
        '''
        Getiitem for EpisodeDataset

        Episode = term used to describe each data set consisting of training and test set i.e. D = (D^train,D^test)
        '''
        # sample task
        taskloader = self.episode_loader[idx] # dataloader class that samples examples form task, mimics x,y ~ P(x,y|task=idx), tasks are modeled by index/label in this problem
        episode_loader = iter(taskloader)
        # get current data set D = (D^{train},D^{test}) as a [n_shot, c, h, w] tensor
        current_dataset_episode = next(episode_loader) # sample batch of size 20 from 600 available in the current task to for a D = (D^{train},D^{test}) split
        return current_dataset_episode # 5,15 data set split D = (D^{train},D^{test})
        #return next(iter(self.episode_loader[idx]))

    def __len__(self):
        return len(self.labels)


class ClassDataset(data.Dataset):
    '''
    Class that holds all the images for a specific class. So it has the 600 images from class=label.
    '''

    def __init__(self, images, label, transform=None):
        """Args:
            images (list of str): each item is a path to an image of the same label
            label (int): the label of all the images
        """
        self.images = images
        self.label = label
        self.transform = transform

    def __getitem__(self, idx):
        image = PILI.open(self.images[idx]).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        return image, self.label

    def __len__(self):
        return len(self.images)


class EpisodicSampler(data.Sampler):

    def __init__(self, total_classes, n_class, n_episode):
        self.total_classes = total_classes
        self.n_class = n_class
        self.n_episode = n_episode

    def __iter__(self):
        for i in range(self.n_episode):
            yield torch.randperm(self.total_classes)[:self.n_class]

    def __len__(self):
        return self.n_episode


def prepare_data(args):

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    train_set = EpisodeDataset(args.data_root, 'train', args.n_shot, args.n_eval,
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

    val_set = EpisodeDataset(args.data_root, 'val', args.n_shot, args.n_eval,
        transform=transforms.Compose([
            transforms.Resize(args.image_size * 8 // 7),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            normalize]))

    test_set = EpisodeDataset(args.data_root, 'test', args.n_shot, args.n_eval,
        transform=transforms.Compose([
            transforms.Resize(args.image_size * 8 // 7),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            normalize]))

    train_loader = data.DataLoader(train_set, num_workers=args.n_workers, pin_memory=args.pin_mem,
        batch_sampler=EpisodicSampler(len(train_set), args.n_class, args.episode))

    val_loader = data.DataLoader(val_set, num_workers=2, pin_memory=False,
        batch_sampler=EpisodicSampler(len(val_set), args.n_class, args.episode_val))

    test_loader = data.DataLoader(test_set, num_workers=2, pin_memory=False,
        batch_sampler=EpisodicSampler(len(test_set), args.n_class, args.episode_val))

    return train_loader, val_loader, test_loader

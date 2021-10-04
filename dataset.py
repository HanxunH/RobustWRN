from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data import Sampler
from util import onehot, rand_bbox
import numpy as np
import torch
import random
import pickle
import logging
import os

# Datasets
available_datasets = ['CIFAR10', 'CIFAR100', 'IMAGENET', 'MNIST']

transform_options = {
    "MNIST": {
        "train_transform": [transforms.ToTensor()],
        "test_transform": [transforms.ToTensor()]
    },
    "IMAGENET": {
        "train_transform": [transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ColorJitter(brightness=0.4,
                                                   contrast=0.4,
                                                   saturation=0.4,
                                                   hue=0.2),
                            transforms.ToTensor()],
        "test_transform": [transforms.Resize(256),
                           transforms.CenterCrop(224),
                           transforms.ToTensor()]},
    "CIFAR10": {
        "train_transform": [transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor()],
        "test_transform": [transforms.ToTensor()]},
    "CIFAR100": {
        "train_transform": [transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomRotation(20),
                            transforms.ToTensor()],
        "test_transform": [transforms.ToTensor()]}
}


class DatasetGenerator():
    def __init__(self, train_batch_size=128, eval_batch_size=256,
                 train_portion=0.5, data_path='data/', dataset_type='CIFAR10',
                 num_of_workers=4, num_of_classes=10, use_cutout=False,
                 use_cutmix=False, use_augmentation=False, use_additional_data=False,
                 additional_data_path='ti_500K_pseudo_labeled.pickle', cutout_length=16):
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.train_portion = train_portion
        self.data_path = data_path
        self.dataset_type = dataset_type
        self.num_of_classes = num_of_classes
        if self.dataset_type not in available_datasets:
            raise('Dataset type %s not implemented' % self.dataset_type)
        self.num_of_workers = num_of_workers
        self.use_cutout = use_cutout
        self.use_cutmix = use_cutmix
        self.use_augmentation = use_augmentation
        self.cutout_length = cutout_length
        self.use_additional_data = use_additional_data
        self.additional_data_path = additional_data_path
        self.data_loaders = self.loadData()
        return

    def getDataLoader(self):
        return self.data_loaders

    def loadData(self):
        train_transform = transform_options[self.dataset_type]['train_transform']
        test_transform = transform_options[self.dataset_type]['test_transform']
        if self.use_cutout and self.dataset_type == 'CIFAR10':
            train_transform[0] = transforms.RandomCrop(32, padding=4, fill=128)

        train_transform = transforms.Compose(train_transform)
        test_transform = transforms.Compose(test_transform)

        if self.use_cutout:
            print('Using Cutout')
            train_transform.transforms.append(Cutout(self.cutout_length))

        if self.dataset_type == 'CIFAR10':
            self.num_of_classes = 10
            if self.use_additional_data:
                train_dataset = SemiSupervisedDataset(root=self.data_path,
                                                      train=True, download=True,
                                                      transform=train_transform,
                                                      aux_data_filename=self.additional_data_path)
                train_batch_sampler = SemiSupervisedSampler(train_dataset.sup_indices,
                                                            train_dataset.unsup_indices,
                                                            self.train_batch_size, 1 - self.train_portion,
                                                            num_batches=int(np.ceil(50000 / self.train_batch_size)))
            else:
                train_dataset = datasets.CIFAR10(root=self.data_path, train=True,
                                                 transform=train_transform, download=True)

            test_dataset = datasets.CIFAR10(root=self.data_path, train=False,
                                            transform=test_transform, download=True)

        elif self.dataset_type == 'CIFAR100':
            self.num_of_classes = 100
            train_dataset = datasets.CIFAR100(root=self.data_path, train=True,
                                              transform=train_transform, download=True)

            test_dataset = datasets.CIFAR100(root=self.data_path, train=False,
                                             transform=test_transform, download=True)

        elif self.dataset_type == 'IMAGENET':
            self.num_of_classes = 1000
            train_dataset = datasets.ImageNet(root=self.data_path, split='train',
                                              transform=train_transform)

            test_dataset = datasets.ImageNet(root=self.data_path, split='val',
                                             transform=test_transform)
        elif self.dataset_type == 'MNIST':
            self.num_of_classes = 10
            train_dataset = datasets.MNIST(root=self.data_path, train=True,
                                           transform=train_transform, download=True)

            test_dataset = datasets.MNIST(root=self.data_path, train=False,
                                          transform=test_transform, download=True)
        else:
            raise('Dataset type %s not implemented' % self.dataset_type)

        if self.use_cutmix:
            train_dataset = CutMix(dataset=train_dataset, num_class=self.num_of_classes)

        data_loaders = {
            'train_total': len(train_dataset),
            'test_total': len(test_dataset)
        }
        print(data_loaders)

        if self.use_additional_data:
            data_loaders['train_dataset'] = DataLoader(dataset=train_dataset,
                                                       batch_sampler=train_batch_sampler,
                                                       pin_memory=True,
                                                       num_workers=self.num_of_workers)
        else:
            data_loaders['train_dataset'] = DataLoader(dataset=train_dataset,
                                                       batch_size=self.train_batch_size,
                                                       shuffle=True,
                                                       pin_memory=True,
                                                       drop_last=True,
                                                       num_workers=self.num_of_workers)

        data_loaders['test_dataset'] = DataLoader(dataset=test_dataset,
                                                  batch_size=self.eval_batch_size,
                                                  shuffle=False,
                                                  pin_memory=True,
                                                  num_workers=self.num_of_workers)
        return data_loaders


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


class CutMix(Dataset):
    def __init__(self, dataset, num_class, num_mix=2, beta=1.0, prob=0.5):
        self.dataset = dataset
        self.num_class = num_class
        self.num_mix = num_mix
        self.beta = beta
        self.prob = prob

    def __getitem__(self, index):
        img, lb = self.dataset[index]
        lb_onehot = onehot(self.num_class, lb)

        for _ in range(self.num_mix):
            r = np.random.rand(1)
            if self.beta <= 0 or r > self.prob:
                continue

            # generate mixed sample
            lam = np.random.beta(self.beta, self.beta)
            rand_index = random.choice(range(len(self)))

            img2, lb2 = self.dataset[rand_index]
            lb2_onehot = onehot(self.num_class, lb2)

            bbx1, bby1, bbx2, bby2 = rand_bbox(img.size(), lam)
            img[:, bbx1:bbx2, bby1:bby2] = img2[:, bbx1:bbx2, bby1:bby2]
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (img.size()[-1] * img.size()[-2]))
            lb_onehot = lb_onehot * lam + lb2_onehot * (1. - lam)

        return img, lb_onehot

    def __len__(self):
        return len(self.dataset)


class SemiSupervisedDataset(Dataset):
    def __init__(self,
                 base_dataset='cifar10',
                 take_amount=None,
                 take_amount_seed=13,
                 add_svhn_extra=False,
                 aux_data_filename=None,
                 add_aux_labels=True,
                 aux_take_amount=None,
                 train=False,
                 **kwargs):
        """A dataset with auxiliary pseudo-labeled data"""

        if base_dataset == 'cifar10':
            self.dataset = datasets.CIFAR10(train=train, **kwargs)
        else:
            raise ValueError('Dataset %s not supported' % base_dataset)
        self.base_dataset = base_dataset
        self.train = train

        if self.train:
            if take_amount is not None:
                rng_state = np.random.get_state()
                np.random.seed(take_amount_seed)
                take_inds = np.random.choice(len(self.sup_indices),
                                             take_amount, replace=False)
                np.random.set_state(rng_state)

                logger = logging.getLogger()
                logger.info('Randomly taking only %d/%d examples from training'
                            ' set, seed=%d, indices=%s',
                            take_amount, len(self.sup_indices),
                            take_amount_seed, take_inds)
                self.targets = self.targets[take_inds]
                self.data = self.data[take_inds]

            self.sup_indices = list(range(len(self.targets)))
            self.unsup_indices = []

            if aux_data_filename is not None:
                aux_path = aux_data_filename
                print("Loading data from %s" % aux_path)
                with open(aux_path, 'rb') as f:
                    aux = pickle.load(f)
                aux_data = aux['data']
                aux_targets = aux['extrapolated_targets']
                orig_len = len(self.data)

                if aux_take_amount is not None:
                    rng_state = np.random.get_state()
                    np.random.seed(take_amount_seed)
                    take_inds = np.random.choice(len(aux_data),
                                                 aux_take_amount, replace=False)
                    np.random.set_state(rng_state)

                    logger = logging.getLogger()
                    logger.info(
                        'Randomly taking only %d/%d examples from aux data'
                        ' set, seed=%d, indices=%s',
                        aux_take_amount, len(aux_data),
                        take_amount_seed, take_inds)
                    aux_data = aux_data[take_inds]
                    aux_targets = aux_targets[take_inds]

                self.data = np.concatenate((self.data, aux_data), axis=0)

                if not add_aux_labels:
                    self.targets.extend([-1] * len(aux_data))
                else:
                    self.targets.extend(aux_targets)
                # note that we use unsup indices to track the labeled datapoints
                # whose labels are "fake"
                self.unsup_indices.extend(
                    range(orig_len, orig_len+len(aux_data)))

            logger = logging.getLogger()
            logger.info("Training set")
            logger.info("Number of training samples: %d", len(self.targets))
            logger.info("Number of supervised samples: %d",
                        len(self.sup_indices))
            logger.info("Number of unsup samples: %d", len(self.unsup_indices))
            logger.info("Label (and pseudo-label) histogram: %s",
                        tuple(
                            zip(*np.unique(self.targets, return_counts=True))))
            logger.info("Shape of training data: %s", np.shape(self.data))

        # Test set
        else:
            self.sup_indices = list(range(len(self.targets)))
            self.unsup_indices = []

            logger = logging.getLogger()
            logger.info("Test set")
            logger.info("Number of samples: %d", len(self.targets))
            logger.info("Label histogram: %s",
                        tuple(
                            zip(*np.unique(self.targets, return_counts=True))))
            logger.info("Shape of data: %s", np.shape(self.data))

    @property
    def data(self):
        return self.dataset.data

    @data.setter
    def data(self, value):
        self.dataset.data = value

    @property
    def targets(self):
        return self.dataset.targets

    @targets.setter
    def targets(self, value):
        self.dataset.targets = value

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        self.dataset.labels = self.targets  # because torchvision is annoying
        return self.dataset[item]

    def __repr__(self):
        fmt_str = 'Semisupervised Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Training: {}\n'.format(self.train)
        fmt_str += '    Root Location: {}\n'.format(self.dataset.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.dataset.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.dataset.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class SemiSupervisedSampler(Sampler):
    """Balanced sampling from the labeled and unlabeled data"""
    def __init__(self, sup_inds, unsup_inds, batch_size, unsup_fraction=0.5,
                 num_batches=None):
        if unsup_fraction is None or unsup_fraction < 0:
            self.sup_inds = sup_inds + unsup_inds
            unsup_fraction = 0.0
        else:
            self.sup_inds = sup_inds
            self.unsup_inds = unsup_inds

        self.batch_size = batch_size
        unsup_batch_size = int(batch_size * unsup_fraction)
        self.sup_batch_size = batch_size - unsup_batch_size

        if num_batches is not None:
            self.num_batches = num_batches
        else:
            self.num_batches = int(
                np.ceil(len(self.sup_inds) / self.sup_batch_size))

        super().__init__(None)

    def __iter__(self):
        batch_counter = 0
        while batch_counter < self.num_batches:
            sup_inds_shuffled = [self.sup_inds[i]
                                 for i in torch.randperm(len(self.sup_inds))]
            for sup_k in range(0, len(self.sup_inds), self.sup_batch_size):
                if batch_counter == self.num_batches:
                    break
                batch = sup_inds_shuffled[sup_k:(sup_k + self.sup_batch_size)]
                if self.sup_batch_size < self.batch_size:
                    batch.extend([self.unsup_inds[i] for i in
                                  torch.randint(high=len(self.unsup_inds),
                                                size=(
                                                    self.batch_size - len(
                                                        batch),),
                                                dtype=torch.int64)])
                # this shuffle operation is very important, without it
                # batch-norm / DataParallel hell ensues
                np.random.shuffle(batch)
                yield batch
                batch_counter += 1

    def __len__(self):
        return self.num_batches

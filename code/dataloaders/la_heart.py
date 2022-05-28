import os
import torch
import numpy as np
import random
from torch.utils.data import Dataset
import h5py
from scipy.ndimage import rotate
import itertools
from torch.utils.data.sampler import Sampler
import sys

# code ref: https://github.com/JunMa11/SegWithDistMap/tree/master/code/dataloaders
# 
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
class LAHeart(Dataset):
    """ LA Dataset """
    def __init__(self, base_dir=None, split='train', num=None, transform=None):
        self._base_dir = base_dir
        self.transform = transform
        self.sample_list = []
        
        if split=='train':
            with open('../data/train.list', 'r') as f:
                self.image_list = f.readlines()
        elif split == 'train80':
            with open('../data/train80.list', 'r') as f:
                self.image_list = f.readlines()
        elif split == 'test':
            with open('../data/test.list', 'r') as f:
                self.image_list = f.readlines()
        self.image_list = [item.replace('\n','') for item in self.image_list]# [:4]
        
        if num is not None:
            
            self.image_list = self.image_list[:num]
        print("total {} samples".format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        #获取路径
        image_name = self.image_list[idx]

        h5f = h5py.File(self._base_dir+"/"+image_name+"/mri_norm2.h5", 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)

        return sample

class CenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape

        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))
        d1 = int(round((d - self.output_size[2]) / 2.))

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]

        return {'image': image, 'label': label}

class ROICrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        tempL = np.nonzero(label)
        minx, maxx = np.min(tempL[0]), np.max(tempL[0])
        miny, maxy = np.min(tempL[1]), np.max(tempL[1])
        minz, maxz = np.min(tempL[2]), np.max(tempL[2])
        # get the center point
        px = (maxx - minx) // 2 + minx
        py = (maxy - miny) // 2 + miny
        pz = (maxz - minz) // 2 + minz
        w, h, d = label.shape
        minx_out = px - self.output_size[0] // 2
        maxx_out = px + self.output_size[0] // 2
        miny_out = py - self.output_size[1] // 2
        maxy_out = py + self.output_size[1] // 2
        minz_out = pz - self.output_size[2] // 2
        maxz_out = pz + self.output_size[2] // 2
        
        if minx_out < 0:
            minx_out = 0
            maxx_out = minx_out + self.output_size[0]
        if maxx_out > d:
            maxx_out = d
            minx_out = maxx_out - self.output_size[0]
        if miny_out < 0:
            miny_out = 0
            maxy_out = miny_out + self.output_size[1]
        if maxy_out > d:
            maxy_out = d
            miny_out = maxy_out - self.output_size[1]
        if minz_out < 0:
            minz_out = 0
            maxz_out = minz_out + self.output_size[2]
        if maxz_out > d:
            maxz_out = d
            minz_out = maxz_out - self.output_size[2]

        label = label[minx_out:maxx_out, miny_out:maxy_out, minz_out:maxz_out]
        image = image[minx_out:maxx_out, miny_out:maxy_out, minz_out:maxz_out]

        return {'image': image, 'label': label}

class RandomScale(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape

        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))
        d1 = int(round((d - self.output_size[2]) / 2.))

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]

        return {'image': image, 'label': label}

class RandomGammaCorrection(object):
    # code source:
    # Chen et al. Multi-task learning for left atrial segmentation on GE-MRI

    def __call__(self, sample):
        # support n-d data
        # :param img:
        # :param mask:
        # :return:
        image, label = sample['image'], sample['label']
        if random.random() < 0.5:
            gamma = random.random() * 1.2 + 0.8  # 0.8-2.0
            print ('gamma: %f', gamma)
            image = image ** (1 / gamma)
            return {'image': image, 'label': label}
        return {'image': image, 'label': label}

class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        return {'image': image, 'label': label}


class RandomRotFlip(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        k = np.random.randint(0, 4) 
        axis_rt=(0,1)
        axis_fp = np.random.randint(0, 1)
        image = np.rot90(image, k, axes=axis_rt)
        label = np.rot90(label, k, axes=axis_rt)

        image = np.flip(image, axis=axis_fp).copy()
        label = np.flip(label, axis=axis_fp).copy()

        return {'image': image, 'label': label}

class RandomFlip(object):
    """
    Crop randomly flip the dataset in a sample
    0 for flipup
    1 for fliplr
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # axis = np.random.randint(0, 1)
        axis = np.random.randint(1, 2)# for lasqs22
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()

        return {'image': image, 'label': label}


class RandomRotation(object):
    """
       Rotate the dataset in a sample
       Args:
       output_size (int): rotated data
       """
    def __init__(self, degrees):
        self.degrees = degrees

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        k = np.random.randint(0, self.degrees)
        image = rotate(image, angle=k, axes=(0, 1), reshape=False, order=1)
        label = rotate(label, angle=k, axes=(0, 1), reshape=False, order=0)
        return {'image': image, 'label': label}
class RandomNoise(object):
    def __init__(self, mu=0, sigma=0.1):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        noise = np.clip(self.sigma * np.random.randn(image.shape[0], image.shape[1], image.shape[2]), -2*self.sigma, 2*self.sigma)
        noise = noise + self.mu
        image = image + noise
        return {'image': image, 'label': label}

class CreateOnehotLabel(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        onehot_label = np.zeros((self.num_classes, label.shape[0], label.shape[1], label.shape[2]), dtype=np.float32)
        for i in range(self.num_classes):
            onehot_label[i, :, :, :] = (label == i).astype(np.float32)
        return {'image': image, 'label': label,'onehot_label':onehot_label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']
        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
        if 'onehot_label' in sample:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long(),
                    'onehot_label': torch.from_numpy(sample['onehot_label']).long()}
        else:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long()}


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size

def iterate_once(iterable):
    return np.random.permutation(iterable) #随机返回一个patch


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)
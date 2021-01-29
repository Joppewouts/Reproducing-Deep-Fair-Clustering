# @Author  : Peizhao Li
# @Contact : peizhaoli05@gmail.com

from PIL import Image
import numpy as np

from data.download_usps import download_usps
from data.download_mnist import download_mnist

import torch
from torch.utils import data
from torchvision import transforms
from os import path

kwargs = {"shuffle": True, "num_workers": 1,
          "pin_memory": True, "drop_last": True}


class digital(data.Dataset):
    # with size in 32x32
    def __init__(self, subset, transform=None, half=False):
        file_dir = "./data/{}.txt".format(subset)
        self.data_dir = open(file_dir).readlines()
        if not path.exists(self.data_dir[0].split()[0].split('/')[2]):

            missing_dir = '/'.join(self.data_dir[0].split()[0].split('/')[:-1])
            print(f'zit in de loop!')
            if subset == 'train_mnist':
                download_mnist(missing_dir)
            if subset == 'train_usps':
                download_usps(missing_dir)
        self.transform = transform
        self.half = half


    def __getitem__(self, index):
        img_dir, label = self.data_dir[index].split()
        img = Image.open(img_dir)
        if self.transform is not None:
            img = self.transform(img)
        label = torch.tensor(np.int64(label)).long()
        if self.half:  # Use float16 tensor for memory efficiency
            img = img.half()
        return img, label

    def __len__(self):
        return len(self.data_dir)


def get_digital(args, subset, transform):
    dataset = digital(subset, transform, half=args.half_tensor)
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=args.bs,
        **kwargs
    )

    return data_loader


def mnist_usps(args):  # TODO: create transformer here and have just 1 get_digital
    transform = transforms.Compose([transforms.ToTensor(),
                                    ])
    train_0 = get_digital(args, "train_mnist", transform)
    train_1 = get_digital(args, "train_usps", transform)
    train_data = [train_0, train_1]

    return train_data


def reverse_mnist(args):
    """
    The protected feature is the source, normal mnist with usps
    or the reversed version of that data
    """
    transform = transforms.Compose([transforms.ToTensor(), ])
    reversed_transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: abs(x - 1.0))])

    train_mnist = get_digital(args, "train_mnist", transform)
    train_mnist_reversed = get_digital(args, "train_mnist", reversed_transform)
    train_data = [train_mnist, train_mnist_reversed]

    return train_data


def office31(args):
    """
    The protected feature is the source, amazon or webcam.
    Pictures are 224x224 with 31 labels.
    """
    # transform=transforms.Compose([transforms.ToTensor(),
    #                                 transforms.Resize(size=(224, 224))
    #                                 ])
    transform = transforms.Compose([ #suggested transform for resnet50 encoder
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    train_0 = get_digital(args, "office31/train_office31_amazon", transform)
    train_1 = get_digital(args, "office31/train_office31_webcam", transform)
    train_data = [train_0, train_1]

    return train_data


def MTFL(args):
    """
    The protected feature is faces with or without glasses.
    Clusters are tested in binary gender classification.
    Pictures are 224x224 with 2 labels.
    """
    transform = transforms.Compose([  # suggested transform for resnet50 encoder
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
            0.229, 0.224, 0.225]),
    ])

    train_0 = get_digital(args, "train_MTFL_with_glasses", transform)
    train_1 = get_digital(args, "train_MTFL_wo_glasses", transform)
    train_data = [train_0, train_1]

    return train_data


get_dataset = {
    'mnist_usps': mnist_usps,
    'reverse_mnist': reverse_mnist,
    'office_31': office31,
    'mtfl': MTFL
}
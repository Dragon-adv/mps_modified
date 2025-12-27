import os.path
import json
import typing
from dataclasses import dataclass

import torch.nn
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision.transforms
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder

from ..model.resnet import ResNet18
from .utils import prepare_result_folder
import global_config


@dataclass
class CommonData:
    encoder: torch.nn.Module
    classifier: torch.nn.Module
    train_set: Dataset
    """Full training set"""
    test_set: Dataset
    train_sets: list[Subset]
    """List of training sets for each client. Each one is a `Subset` of `train_set`."""
    train_loaders: list[DataLoader]
    test_loader: DataLoader
    class_num: int
    augmentation: list
    result_folder: str
    feature_dim: int
    @property
    def client_num(self) -> int: return len(self.train_loaders)


def make_common_data(
    dataset: str,
    batch_size: int,
    split_filepath: str,
    name: str,
    start_time,
    record_text: str,
    aug: str = 'default'
) -> CommonData:
    encoder = ResNet18()
    encoder.fc = typing.cast(typing.Any, torch.nn.Identity())
    if aug == 'default':
        augmentation = [
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomResizedCrop(size=32, scale=(0.64, 1.0), ratio=(1.0, 1.0)),
        ]
    elif aug == 'weak1':
        augmentation = [
            torchvision.transforms.RandomHorizontalFlip(),
        ]
    elif aug == 'weak2':
        augmentation = [
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomResizedCrop(size=32, scale=(0.93, 1.0), ratio=(1.0, 1.0)),
        ]
    elif aug == 'none':
        augmentation = []
    if dataset == 'cifar10_lt':
        train_set = CIFAR10(
            root=global_config.TORCHVISION_ROOT,
            train=True,
            transform=torchvision.transforms.Compose([
                *augmentation,
                torchvision.transforms.ToTensor()
            ]),
            download=True
        )
        test_set = CIFAR10(
            root=global_config.TORCHVISION_ROOT,
            train=False,
            transform=torchvision.transforms.ToTensor()
        )
        class_num = 10
    elif dataset == 'cifar100_lt':
        train_set = CIFAR100(
            root=global_config.TORCHVISION_ROOT,
            train=True,
            transform=torchvision.transforms.Compose([
                *augmentation,
                torchvision.transforms.ToTensor()
            ]),
            download=True
        )
        test_set = CIFAR100(
            root=global_config.TORCHVISION_ROOT,
            train=False,
            transform=torchvision.transforms.ToTensor()
        )
        class_num = 100
    elif dataset == 'cinic10_lt':
        train_set = ImageFolder(
            root=os.path.join(global_config.CINIC10_ROOT, 'train'),
            transform=torchvision.transforms.Compose([
                *augmentation,
                torchvision.transforms.ToTensor()
            ])
        )
        test_set = ImageFolder(
            root=os.path.join(global_config.CINIC10_ROOT, 'test'),
            transform=torchvision.transforms.ToTensor()
        )
        class_num = 10
    else:
        raise NotImplementedError(f'Dataset {dataset} is not supported.')
    feature_dim = 512
    classifier = torch.nn.Linear(feature_dim, class_num)
    train_sets = make_client_subsets(train_set, split_filepath)
    train_loaders = [
        DataLoader(
            subset, batch_size=batch_size, shuffle=True,
            num_workers=global_config.NUM_WORKERS,
            persistent_workers=global_config.PERSISTENT_WORKERS
        )
        for subset in train_sets
    ]
    if dataset == 'cinic10_lt':
        test_loader = DataLoader(test_set, batch_size=global_config.INFERENCE_BATCH_SIZE, num_workers=global_config.NUM_WORKERS)
    else:
        test_loader = DataLoader(test_set, batch_size=global_config.INFERENCE_BATCH_SIZE)
    train_indices: list[int] = []
    for subset in train_sets:
        train_indices.extend(subset.indices)
    run_folder = prepare_result_folder(
        name=name,
        start_time=start_time,
        record_text=record_text
    )
    return CommonData(
        encoder=encoder,
        classifier=classifier,
        train_set=train_set,
        test_set=test_set,
        train_sets=train_sets,
        train_loaders=train_loaders,
        test_loader=test_loader,
        class_num=class_num,
        feature_dim=feature_dim,
        augmentation=augmentation,
        result_folder=run_folder
    )


def make_client_subsets(dataset, split_filepath: str):
    with open(split_filepath) as file:
        results: list[list[int]] = json.load(file)
    return [Subset(dataset, indices) for indices in results]

import typing
import os.path

import torch
from torch.utils.data import DataLoader
import torchvision.transforms
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder

from ..commons.evaluate import evaluate_metric
from ..model.resnet import ResNet18
from .utils import RunRecord
import global_config


@RunRecord.record
def run(
    *,
    dataset: str,
    device,
    # ------
    model_ckpt: 'str | None' = None,  # checkpoint path of the global model
    encoder: 'torch.nn.Module | None' = None,  # encoder model to use, if None, will be loaded from model_ckpt
    classifier: 'torch.nn.Module | None' = None,  # classifier model to use, if None, will be loaded from model_ckpt
    run_record=RunRecord(print_type_args=['encoder', 'classifier'])
):
    print(run_record)
    device = torch.device(device)
    if dataset == 'cifar10_lt':
        class_num = 10
        test_set = CIFAR10(
            root=global_config.TORCHVISION_ROOT,
            train=False,
            transform=torchvision.transforms.ToTensor()
        )
    elif dataset == 'cifar100_lt':
        class_num = 100
        test_set = CIFAR100(
            root=global_config.TORCHVISION_ROOT,
            train=False,
            transform=torchvision.transforms.ToTensor()
        )
    elif dataset == 'cinic10_lt':
        class_num = 10
        test_set = ImageFolder(
            root=os.path.join(global_config.CINIC10_ROOT, 'test'),
            transform=torchvision.transforms.ToTensor()
        )
    else:
        raise ValueError(f'Unknown dataset: {dataset}')
    if dataset == 'cinic10_lt':
        test_loader = DataLoader(test_set, batch_size=global_config.INFERENCE_BATCH_SIZE, num_workers=global_config.NUM_WORKERS)
    else:
        test_loader = DataLoader(test_set, batch_size=global_config.INFERENCE_BATCH_SIZE)

    if model_ckpt is not None:
        checkpoint = torch.load(model_ckpt, map_location='cpu')
        feature_dim = 512
        encoder = ResNet18()
        encoder.fc = typing.cast(typing.Any, torch.nn.Identity())
        encoder.load_state_dict(checkpoint['encoder'])
        classifier = torch.nn.Linear(feature_dim, class_num, bias='bias' in checkpoint['classifier'])
        classifier.load_state_dict(checkpoint['classifier'])
    assert encoder is not None and classifier is not None

    acc = evaluate_metric(
        model=torch.nn.Sequential(encoder, classifier),
        test_loader=test_loader,
        device=device,
        back_to_cpu=True
    )['acc']
    print(f'Test accuracy: {acc:.4f}')

    return {
        'acc': acc
    }

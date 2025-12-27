import typing
import os.path
import copy

import torch
from torch.utils.data import DataLoader
import torchvision.transforms
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder

from ..sfd.finetune import finetune_classifier, extract_test_z_y
from ..model.resnet import ResNet18
from .utils import RunRecord, prepare_result_folder
import global_config


@RunRecord.record
def run(
    *,
    dataset: str,
    name: str = 'SFD_finetune',
    device,
    # ------
    syn_dataset: 'list[dict[str, typing.Any]] | None' = None,
    safs_ckpt: 'str | None' = None,
    # ------
    learned_ckpt: str,
    epochs: int,
    lr: float,
    batch_size: int,
    weight_decay: float,
    save_to_ckpt: bool = True,
    run_record=RunRecord(print_type_args=['syn_dataset'])
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

    learned_checkpoint = torch.load(learned_ckpt, map_location='cpu')
    feature_dim = 512
    encoder = ResNet18()
    encoder.fc = typing.cast(typing.Any, torch.nn.Identity())
    encoder.load_state_dict(learned_checkpoint['encoder'])
    classifier = torch.nn.Linear(feature_dim, class_num)
    with torch.no_grad():
        classifier.weight.copy_(learned_checkpoint['classifier']['weight'])
        if 'bias' in learned_checkpoint['classifier']:
            classifier.bias.copy_(learned_checkpoint['classifier']['bias'])

    test_z, test_y = extract_test_z_y(encoder, test_loader=test_loader, device=device)

    if syn_dataset is None:
        assert safs_ckpt is not None, 'safs_ckpt must be provided if syn_dataset is None'
        syn_dataset = torch.load(safs_ckpt, map_location='cpu')
    assert syn_dataset is not None

    syn_z = torch.concat([class_data['synthetic_features'] for class_data in syn_dataset])
    syn_y = torch.concat([
        torch.full(
            (class_data['synthetic_features'].size(0),),
            class_data['class_index'],
            dtype=torch.int64
        )
        for class_data in syn_dataset
    ])

    new_classifier = copy.deepcopy(classifier)
    finetune_classifier(
        classifier=new_classifier,
        syn_z=syn_z,
        syn_y=syn_y,
        test_z=test_z,
        test_y=test_y,
        device=device,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        weight_decay=weight_decay
    )

    if save_to_ckpt:
        result_folder = prepare_result_folder(name=name, start_time=run_record.start_time, record_text=str(run_record))
        finetune_checkpoint = learned_checkpoint.copy()
        finetune_checkpoint['classifier'] = new_classifier.state_dict()
        torch.save(
            finetune_checkpoint,
            os.path.join(result_folder, 'finetuned.pt')
        )
    else:
        result_folder = None

    return {
        'result_folder': result_folder,
        'finetuned_classifier': new_classifier,
    }

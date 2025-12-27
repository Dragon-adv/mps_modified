import os.path
import json

import numpy as np
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder

import global_config


def make_long_tailed_indices(
    dataset_name: str,
    imbalance_factor
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """
    :return: indices, labels, class_sizes - The indices and labels of the samples.
    """
    # Load dataset
    if dataset_name == 'cifar10_lt':
        dataset = CIFAR10(global_config.TORCHVISION_ROOT, train=True)
        num_classes = 10
    elif dataset_name == 'cifar100_lt':
        dataset = CIFAR100(global_config.TORCHVISION_ROOT, train=True)
        num_classes = 100
    elif dataset_name == 'cinic10_lt':
        dataset = ImageFolder(os.path.join(global_config.CINIC10_ROOT, 'train'))
        num_classes = 10
    else:
        raise ValueError("Invalid dataset_name")

    # Get dataset labels
    targets = np.array(dataset.targets)
    num_max = len(targets) // num_classes  # Number of samples per class (under balanced condition)
    assert num_max == max(np.bincount(targets))

    # Calculate the number of samples for each class to create a long-tailed distribution based on imbalance_factor
    # The number of samples for the i-th class (i from 0 to num_classes - 1) is n_i = n_max * (imbalance_factor ^ (-i / (num_classes - 1)))
    class_sizes = [int(num_max * (imbalance_factor ** (-i / (num_classes - 1)))) for i in range(num_classes)]
    print(f'Number of samples per class: {class_sizes}')

    # Get the sample indices for each class
    indices = []
    labels = []
    for class_idx, count in enumerate(class_sizes):
        # Get the indices of all samples belonging to the current class
        class_indices = np.where(targets == class_idx)[0]
        # Randomly select 'count' samples
        sampled_indices = np.random.choice(class_indices, count, replace=False)
        indices.extend(sampled_indices)
        labels.extend([class_idx] * count)

    return np.array(indices), np.array(labels), class_sizes


def run(
    dataset_name: str,
    imbalance_factor: int,
    output_folder: str
):
    indices, labels, class_sizes = make_long_tailed_indices(dataset_name, imbalance_factor)
    output_filepath = os.path.join(output_folder, f'{dataset_name}_lt,global,if={imbalance_factor}.json')
    # Prevent accidental overwriting of existing files
    if os.path.exists(output_filepath):
        raise FileExistsError(f'File {output_filepath} already exists. Please choose a different output folder or delete the existing file.')
    # Ensure the output folder exists, create it if it does not
    os.makedirs(output_folder, exist_ok=True)
    # save to the output JSON file
    with open(output_filepath, 'w') as file:
        json.dump({
            'indices': indices.tolist(),
            'labels': labels.tolist(),
            'dataset_name': dataset_name,
            'imbalance_factor': imbalance_factor,
            'class_sizes': class_sizes
        }, file, indent=4)
    print(f'Long-tailed dataset {dataset_name} with IF={imbalance_factor} saved to {output_filepath}')

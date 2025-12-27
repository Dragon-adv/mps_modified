import os.path
import json

import numpy as np
from fedbox.utils.data import split_dirichlet_label


def run(
    long_tailed_json_path: str,
    client_num: int,
    dirichlet_alpha: float,
    output_folder: str
):
    with open(long_tailed_json_path, 'r') as file:
        long_tailed_dataset: dict = json.load(file)
    indices = np.array(long_tailed_dataset['indices'])
    labels = np.array(long_tailed_dataset['labels'])
    dataset_name: str = long_tailed_dataset['dataset_name']
    imbalance_factor: int = long_tailed_dataset['imbalance_factor']
    splitting = split_dirichlet_label(
        indices,
        labels,
        client_num=client_num,
        alpha=dirichlet_alpha,
    )

    output_filepath = os.path.join(output_folder, f'{dataset_name}_lt,dir_label,client={client_num},if={imbalance_factor},alpha={dirichlet_alpha}.json')
    # Prevent accidental overwriting of existing files
    if os.path.exists(output_filepath):
        raise FileExistsError(f'File {output_filepath} already exists. Please choose a different output folder or delete the existing file.')
    # Ensure the output folder exists, create it if it does not
    os.makedirs(output_folder, exist_ok=True)
    # Save sample indices of each client to the output JSON file
    with open(output_filepath, 'w') as file:
        json.dump([indices.tolist() for indices, _ in splitting], file, indent=4)
    print(
        f'Fed-LT dataset {dataset_name}, IF={imbalance_factor}, client_num={client_num}, alpha={dirichlet_alpha}\n'
        f'Sample indices of each client saved to {output_filepath}'
    )

import typing
import os.path

import torch

from ..sfd.safs import feature_synthesis, make_syn_nums
from ..sfd.nn import RFF, MeanCovAligner
from .utils import RunRecord, prepare_result_folder


@RunRecord.record
def run(
    *,
    dataset: str,
    name: str = 'SFD_safs',
    device,
    # ------
    stats_ckpt: 'str | None' = None,  # checkpoint path of global statistics and RF model
    rf_model: 'RFF | None' = None,  # RF model to use, if None, will be loaded from stats_ckpt
    global_stats: 'dict[str, typing.Any] | None' = None,
    # ------
    syn_nums: 'list[int] | None' = None,  # number of synthetic features per class, if None, will be calculated
    max_syn_num: 'int | None' = None,
    min_syn_num: 'int | None' = None,
    # ------
    steps: int,
    lr: float,
    target_cov_eps: float,
    run_record=RunRecord(print_type_args=['rf_model', 'global_stats'])
):
    print(run_record)
    device = torch.device(device)
    feature_dim = 512
    if dataset == 'cifar10_lt' or dataset == 'cinic10_lt':
        class_num = 10
    elif dataset == 'cifar100_lt':
        class_num = 100
    else:
        raise ValueError(f'Unknown dataset: {dataset}')

    if stats_ckpt is not None:
        checkpoint = torch.load(stats_ckpt, map_location='cpu')
        rf_args = checkpoint['rf_args']
        rf_model = RFF(
            d=feature_dim,
            device=device,
            D=rf_args['rf_dim'],
            gamma=rf_args['rbf_gamma'],
            rf_type=rf_args['rf_type']
        )
        rf_model.load_state_dict(checkpoint['rf_model'])
        global_stats = checkpoint['global_stats']
    assert rf_model is not None and global_stats is not None, \
        'rf_model and global_stats must be provided if stats_ckpt is None'
    class_rf_means: list[torch.Tensor] = global_stats['class_rf_means']
    class_means: list[torch.Tensor] = global_stats['class_means']
    class_covs: list[torch.Tensor] = global_stats['class_covs']


    if syn_nums is None:
        assert max_syn_num is not None and min_syn_num is not None, \
            'max_syn_num and min_syn_num must be provided if syn_nums is None'
        syn_nums = make_syn_nums(
            global_stats['sample_per_class'].tolist(),
            max_num=max_syn_num,
            min_num=min_syn_num
        )
    assert min(syn_nums) > feature_dim, 'minimum syn_num must be greater than feature_dim'

    aligners: list[MeanCovAligner] = []
    for c in range(class_num):
        aligners.append(
            MeanCovAligner(
                target_mean=class_means[c],
                target_cov=class_covs[c],
                target_cov_eps=target_cov_eps
            )
        )

    class_syn_datasets = feature_synthesis(
        feature_dim=feature_dim,
        class_num=class_num,
        device=device,
        aligners=aligners,
        rf_model=rf_model,
        class_rf_means=class_rf_means,
        steps=steps,
        lr=lr,
        syn_num_per_class=syn_nums,
    )
    
    result_folder = prepare_result_folder(name=name, start_time=run_record.start_time, record_text=str(run_record))
    torch.save(class_syn_datasets, os.path.join(result_folder, 'class_syn_datasets.pt'))

    return {
        'result_folder': result_folder,
        'class_syn_datasets': class_syn_datasets,
    }

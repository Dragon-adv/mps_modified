import torch
from tqdm import tqdm

from .nn import RFF, MeanCovAligner


def feature_synthesis(
    *,
    feature_dim: int,
    class_num: int,
    device,
    aligners: list[MeanCovAligner],
    rf_model: RFF,
    class_rf_means: list[torch.Tensor],
    steps: int,
    lr: float,
    syn_num_per_class: list[int],
):
    """
    :return: list of dicts, each dict contains:

        - 'class_index' (int): index of the class
        - 'synthetic_raw_features' (torch.Tensor): raw features for the class
        - 'synthetic_features' (torch.Tensor): aligned features for the class
    """
    memory_banks: list[torch.Tensor] = []  # synthetic raw features for each class
    class_syn_z: list[torch.Tensor] = []  # synthetic features for each class
    rf_model.to(device)
    INPUT_COV_EPS = 1e-6
    for c in tqdm(range(class_num), desc='train banks', leave=False):
        aligner = aligners[c].to(device)
        torch.cuda.empty_cache()
        # train memory_bank (synthetic raw feature) for class c
        syn_zc_raw = train_memory_bank(
            feature_dim=feature_dim,
            syn_num=syn_num_per_class[c],
            device=device,
            aligner=aligners[c],
            rf_model=rf_model,
            real_rf_mean=class_rf_means[c],
            steps=steps,
            lr=lr,
            input_cov_eps=INPUT_COV_EPS,
        )
        memory_banks.append(syn_zc_raw)
        torch.cuda.empty_cache()

        # make synthetic dataset for class c
        syn_zc = aligner.forward(syn_zc_raw.to(device), input_cov_eps=INPUT_COV_EPS).cpu()
        assert torch.isfinite(syn_zc).all(), f'syn_zc is not finite: {syn_zc}'
        class_syn_z.append(syn_zc)

        aligner.cpu()
        torch.cuda.empty_cache()
    rf_model.cpu()
    torch.cuda.empty_cache()

    return [
        {
            'class_index': c,
            'synthetic_raw_features': memory_banks[c],
            'synthetic_features': class_syn_z[c],
        }
        for c in range(class_num)
    ]


def train_memory_bank(
    *,
    feature_dim: int,
    syn_num: int,
    device: 'torch.device | str',
    aligner: MeanCovAligner,
    rf_model: RFF,
    real_rf_mean: torch.Tensor,
    steps: int,
    lr: float,
    input_cov_eps: float,
) -> torch.Tensor:
    """
    :return: trained memory bank (torch.Tensor)
    """
    torch.cuda.empty_cache()
    memory_bank = torch.nn.Parameter(
        torch.randn((syn_num, feature_dim), device=device),
        requires_grad=True
    )
    real_rf_mean = real_rf_mean.to(device)
    optimizer = torch.optim.SGD([memory_bank], lr=lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    progress_bar = tqdm(range(steps), desc='train a bank', leave=False)
    for step in progress_bar:
        syn_z_raw = memory_bank
        syn_z = aligner.forward(syn_z_raw, input_cov_eps=input_cov_eps)
        assert torch.isfinite(syn_z).all(), f'syn_z is not finite: {syn_z}'
        syn_rf_mean = rf_model.forward(syn_z).mean(dim=0)
        mmd_loss = (syn_rf_mean - real_rf_mean).abs().sum()
        # range regularization for features after ReLU activation,
        # encourage the synthetic features to be non-negative
        range_reg = -torch.min(torch.tensor(0.0), syn_z).sum(dim=1).mean(dim=0)
        loss = mmd_loss + range_reg
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
    trained_memory_bank = memory_bank.detach().cpu()
    torch.cuda.empty_cache()
    return trained_memory_bank


def make_syn_nums(class_sizes: list[int], max_num: int, min_num: int) -> list[int]:
    """
    Generate synthetic numbers for each class based on their size order.
    The smallest class gets `max_num`, and the largest class gets `min_num`.
    If all classes have the same size, they all get `max_num`.

    :param class_sizes: list of class sizes, only consider the order, not the actual values
    :param max_num: maximum synthetic number for the smallest class
    :param min_num: minimum synthetic number for the largest class
    :return: list of synthetic numbers for each class
    """
    assert len(class_sizes) > 0, 'class_sizes must not be empty'
    assert max_num >= min_num, 'max_num must be greater than or equal to min_num'

    unique_sorted_sizes = sorted(list(set(class_sizes)))
    num_unique_sizes = len(unique_sorted_sizes)
    size_to_syn_num_map: dict[int, int] = {}

    if num_unique_sizes == 1:
        size_to_syn_num_map[unique_sorted_sizes[0]] = max_num
    else:
        scaling_denominator = float(num_unique_sizes - 1)
        for rank, size in enumerate(unique_sorted_sizes):
            syn_num_float = max_num - (rank / scaling_denominator) * (max_num - min_num)
            size_to_syn_num_map[size] = int(round(syn_num_float))
    result_syn_nums = [size_to_syn_num_map[size] for size in class_sizes]
    return result_syn_nums

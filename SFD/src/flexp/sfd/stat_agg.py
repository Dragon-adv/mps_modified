import torch
import torch.nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .nn import RFF


@torch.no_grad()
def compute_local_stats(
    *,
    encoder: torch.nn.Module,
    rf_model: RFF,
    class_num: int,
    feature_dim: int,
    device,
    train_set,
    epochs: int = 1,
    batch_size: int,
    num_workers: int,
):
    """
    :return: class_means, class_outers, class_rf_means, sample_per_class
    """
    class_means: list[torch.Tensor] = []
    class_outers: list[torch.Tensor] = []  # mean of outer products
    class_rf_means: list[torch.Tensor] = []
    torch.cuda.empty_cache()
    encoder.eval().to(device)
    rf_model.to(device)
    loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers)
    zs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    rfs: list[torch.Tensor] = []
    for epoch in tqdm(range(epochs), desc=f'extract local feature', leave=False):
        for x, y in tqdm(loader, desc=f'epoch {epoch}', leave=False):
            x = x.to(device)
            feature = encoder(x)
            rf = rf_model.forward(feature)
            zs.append(feature.cpu())
            ys.append(y.cpu())
            rfs.append(rf.cpu())
    encoder.cpu()
    rf_model.cpu()
    torch.cuda.empty_cache()
    z = torch.concat(zs, dim=0)  # shape (sample_num * epoch, feature_dim)
    y = torch.concat(ys, dim=0)  # shape (sample_num * epoch,)
    rf = torch.concat(rfs, dim=0)
    sample_per_class = y.bincount(minlength=class_num)
    for c in range(class_num):
        if sample_per_class[c] == 0:
            class_mean = torch.zeros(feature_dim)
            class_outer = torch.zeros(feature_dim, feature_dim)
            class_rf_mean = torch.zeros(rf_model.D)
        else:
            class_indices = (y == c)
            class_features = z[class_indices]
            class_mean = class_features.mean(dim=0)
            class_f = class_features.to(torch.float64)
            class_outer = (torch.matmul(class_f.t(), class_f) / class_features.size(0)).to(class_features.dtype)  # outer product mean
            class_rf_mean = rf[class_indices].mean(dim=0)
        class_means.append(class_mean)
        class_outers.append(class_outer)
        class_rf_means.append(class_rf_mean)
    return {
        'class_means': class_means,
        'class_outers': class_outers,
        'class_rf_means': class_rf_means,
        'sample_per_class': sample_per_class
    }


def compute_global_stats(
    client_responses: list[dict[str, torch.Tensor]],
    class_num: int,
    feature_dim: int,
    rf_dim: int,
):
    """
    :param client_responses: list of responses from clients, each response contains 'class_means', 'class_outers', 'class_rf_means', 'sample_per_class'
    :return: class_means, class_outers, class_covs, class_rf_means, sample_per_class
    """
    assert len(client_responses) > 0, 'No client stats responses to aggregate.'
    global_class_means = [torch.zeros(feature_dim, dtype=torch.float64) for _ in range(class_num)]
    global_class_outers = [torch.zeros((feature_dim, feature_dim), dtype=torch.float64) for _ in range(class_num)]
    global_class_covs = [torch.zeros((feature_dim, feature_dim), dtype=torch.float64) for _ in range(class_num)]
    global_class_rf_means = [torch.zeros(rf_dim, dtype=torch.float64) for _ in range(class_num)]
    global_sample_per_class = torch.zeros(class_num)

    for response in client_responses:
        for c in range(class_num):
            n_kc = response['sample_per_class'][c]
            if n_kc == 0:
                continue
            global_class_means[c] += n_kc * response['class_means'][c].to(torch.float64)
            global_class_outers[c] += n_kc * response['class_outers'][c].to(torch.float64)
            global_class_rf_means[c] += n_kc * response['class_rf_means'][c].to(torch.float64)
        global_sample_per_class += response['sample_per_class']

    for c in range(class_num):
        n_c = global_sample_per_class[c]
        ori_dtype = client_responses[0]['class_means'][0].dtype
        if n_c > 0:
            global_class_means[c] /= n_c
            global_class_outers[c] /= n_c
            global_class_rf_means[c] /= n_c
            global_class_covs[c] = global_class_outers[c] - torch.outer(global_class_means[c], global_class_means[c])
        global_class_means[c] = global_class_means[c].to(ori_dtype)
        global_class_outers[c] = global_class_outers[c].to(ori_dtype)
        global_class_covs[c] = global_class_covs[c].to(ori_dtype)
        global_class_rf_means[c] = global_class_rf_means[c].to(ori_dtype)
    return {
        'class_means': global_class_means,
        'class_outers': global_class_outers,
        'class_covs': global_class_covs,
        'class_rf_means': global_class_rf_means,
        'sample_per_class': global_sample_per_class
    }

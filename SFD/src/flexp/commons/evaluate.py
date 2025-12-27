import typing

import torch
import torch.nn
from sklearn.metrics import accuracy_score
from tqdm import tqdm


@torch.no_grad()
def evaluate_metric(
    model: torch.nn.Module,
    test_loader: typing.Iterable[tuple[torch.Tensor, torch.Tensor]],
    device: 'torch.device | str',
    back_to_cpu: bool = True
) -> dict[str, float]:
    """
    :return: {'acc': float}
    """
    model.to(device).eval()
    target_list: list[torch.Tensor] = []
    pred_list: list[torch.Tensor] = []
    for batch in tqdm(test_loader, desc='eval test', leave=False):
        x, y = batch
        pred = model(x.to(device)).argmax(dim=1).cpu()
        pred_list.append(pred)
        target_list.append(y)
    if back_to_cpu:
        model.cpu()
    target = torch.concat(target_list)
    pred = torch.concat(pred_list)
    acc = accuracy_score(target, pred)
    return {
        'acc': float(acc)
    }

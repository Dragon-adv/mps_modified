import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from ..commons.evaluate import evaluate_metric
from fedbox.utils.training import Recorder
import global_config


@torch.no_grad()
def extract_test_z_y(
    encoder: torch.nn.Module,
    test_loader,
    device,
):
    """
    :return: test_z, test_y
    """
    torch.cuda.empty_cache()
    encoder.eval().to(device)
    zs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    for x, y in tqdm(test_loader, desc='extract test_z_y', leave=False):
        x = x.to(device)
        with torch.no_grad():
            feature = encoder(x)
        zs.append(feature.cpu())
        ys.append(y)
    encoder.cpu()
    torch.cuda.empty_cache()
    return torch.concat(zs, dim=0), torch.concat(ys, dim=0)


def finetune_classifier(
    *,
    classifier: torch.nn.Module,
    syn_z: torch.Tensor,
    syn_y: torch.Tensor,
    test_z: torch.Tensor,
    test_y: torch.Tensor,
    device,
    epochs: int,
    lr: float,
    batch_size: int,
    weight_decay: float,
):
    classifier.to(device).train()
    test_zy_loader = DataLoader(TensorDataset(test_z, test_y), batch_size=global_config.INFERENCE_BATCH_SIZE)
    optimizer = torch.optim.SGD(classifier.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    recorder = Recorder(higher_better=True)
    syn_loader = DataLoader(TensorDataset(syn_z, syn_y), batch_size=batch_size, shuffle=True)

    acc = evaluate_metric(
        model=classifier,
        test_loader=test_zy_loader,
        device=device,
        back_to_cpu=False
    )['acc']
    tqdm.write(f'before finetune, test acc: {acc}')

    current_iter = 0
    progress_bar = tqdm(range(epochs), desc='finetune classifier', leave=False)
    for epoch in progress_bar:
        for z, y in tqdm(syn_loader, desc=f'epoch {epoch}', leave=False):
            z, y = z.to(device), y.to(device)
            logit = classifier(z)
            loss = torch.nn.functional.cross_entropy(logit, y)
            assert torch.isfinite(loss), f'loss is not finite: {loss}'
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            current_iter += 1
        acc = evaluate_metric(
            model=classifier, test_loader=test_zy_loader,
            device=device, back_to_cpu=False
        )['acc']
        is_best = recorder.update(acc, iter=current_iter, epoch=epoch)
        tqdm.write(f'epoch {epoch}, iter {current_iter}, acc: {acc:.4g}, is_best: {is_best}')
    tqdm.write(f'Best acc: {recorder.best_metric}, epoch {recorder["epoch"]}, iter {recorder["iter"]}')
    classifier.cpu()

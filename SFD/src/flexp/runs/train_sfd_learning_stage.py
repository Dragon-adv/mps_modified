import copy

import torch

from ..sfd import SFDClient, SFDServer
from .commons import make_common_data
from .utils import RunRecord, keep_with_confirm


@RunRecord.record
def run(
    *,
    dataset: str,
    split_filepath: str,
    name: str = 'SFD_learn',
    device,
    a_ce_gamma: float,
    beta_pi: float,
    scl_weight_start: float = 0.1,
    global_rounds: int = 200,
    participate_ratio: float = 1.0,
    local_epochs: int = 5,
    batch_size: int = 64,
    lr: float = 0.01,
    momentum: float = 0.9,
    weight_decay: float = 1e-5,
    scl_temperature: float = 0.07,
    run_record=RunRecord()
):
    print(run_record)
    device = torch.device(device)
    com = make_common_data(
        dataset=dataset,
        batch_size=batch_size,
        split_filepath=split_filepath,
        name=name,
        start_time=run_record.start_time,
        record_text=str(run_record)
    )
    assert isinstance(com.classifier, torch.nn.Linear)
    class_num, feature_dim = com.classifier.weight.shape
    assert class_num == com.class_num, f'Expected class_num {com.class_num}, but got {class_num}'
    assert feature_dim == com.feature_dim, f'Expected feature_dim {com.feature_dim}, but got {feature_dim}'
    com.classifier = torch.nn.Linear(feature_dim, class_num, bias=False)
    # projector in contrastive learning
    assert feature_dim == 512
    projector = torch.nn.Sequential(
        torch.nn.Linear(feature_dim, 256),
        torch.nn.BatchNorm1d(256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 128)
    )

    clients = [
        SFDClient(
            id=i,
            encoder=copy.deepcopy(com.encoder),
            classifier=copy.deepcopy(com.classifier),
            projector=copy.deepcopy(projector),
            train_loader=com.train_loaders[i],
            train_set=com.train_sets[i],
            class_num=class_num,
            feature_dim=feature_dim,
            local_epochs=local_epochs,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            scl_temperature=scl_temperature,
            a_ce_gamma=a_ce_gamma,
            beta_pi=beta_pi,
            device=device
        ) for i in range(com.client_num)
    ]
    server = SFDServer(
        clients=clients.copy(),
        encoder=copy.deepcopy(com.encoder),
        classifier=copy.deepcopy(com.classifier),
        projector=copy.deepcopy(projector),
        test_loader=com.test_loader,
        class_num=class_num,
        feature_dim=feature_dim,
        global_rounds=global_rounds,
        participate_ratio=participate_ratio,
        device=device,
        scl_weight_start=scl_weight_start,
        result_folder=com.result_folder
    )
    try:
        server.fit()
    except KeyboardInterrupt:
        keep_with_confirm(com.result_folder)
        raise
    return {
        'result_folder': com.result_folder,
    }

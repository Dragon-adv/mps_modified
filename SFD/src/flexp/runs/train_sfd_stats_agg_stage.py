import copy
import typing
import os.path

import torch

from ..sfd import SFDClient, SFDServer
from .commons import make_common_data
from .utils import RunRecord, keep_with_confirm


@RunRecord.record
def run(
    *,
    dataset: str,
    split_filepath: str,
    name: str = 'SFD_stats',
    device,
    learned_ckpt: str,  # checkpoint file path of the learned model
    save_to_ckpt: bool,
    rf_dim: int = 3000,
    rbf_gamma: float = 0.01,
    run_record=RunRecord()
):
    print(run_record)
    device = torch.device(device)
    com = make_common_data(
        dataset=dataset,
        split_filepath=split_filepath,
        name=name,
        start_time=run_record.start_time,
        aug='weak2',
        record_text=str(run_record),
        batch_size=1,  # unused in this stage
    )
    assert isinstance(com.classifier, torch.nn.Linear)
    class_num, feature_dim = com.classifier.weight.shape
    assert class_num == com.class_num, f'Expected class_num {com.class_num}, but got {class_num}'
    assert feature_dim == com.feature_dim, f'Expected feature_dim {com.feature_dim}, but got {feature_dim}'
    com.classifier = torch.nn.Linear(feature_dim, class_num, bias=False)
    projector = torch.nn.Identity()  # no need for projector in this stage
    checkpoint = torch.load(learned_ckpt, map_location='cpu')
    com.encoder.load_state_dict(checkpoint['encoder'])
    com.classifier.load_state_dict(checkpoint['classifier'])
    
    if not save_to_ckpt:
        # remove the created result_folder
        import shutil
        shutil.rmtree(com.result_folder, ignore_errors=True)

    UNUSED: typing.Any = None  # placeholder for unused parameters
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
            device=device,
            # --- unused in this stage ---
            local_epochs=UNUSED,
            lr=UNUSED,
            momentum=UNUSED,
            weight_decay=UNUSED,
            a_ce_gamma=UNUSED,
            beta_pi=UNUSED,
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
        device=device,
        result_folder=com.result_folder,
        # --- unused in this stage ---
        global_rounds=UNUSED,
        scl_weight_start=UNUSED
    )

    try:
        results = server.get_global_statistics(
            rf_dim=rf_dim,
            rbf_gamma=rbf_gamma
        )
        global_stats = results['global_stats']
        rf_model = results['rf_model']
        rf_args = results['rf_args']
        if save_to_ckpt:
            torch.save({
                'global_stats': global_stats,
                'rf_model': rf_model.state_dict(),
                'rf_args': rf_args,
            }, os.path.join(com.result_folder, 'stats_stage.pt'))
    except KeyboardInterrupt:
        if save_to_ckpt:
            keep_with_confirm(com.result_folder)
        raise

    return {
        'result_folder': com.result_folder,
        'global_stats': global_stats,
        'rf_model': rf_model,
        'rf_args': rf_args,
    }

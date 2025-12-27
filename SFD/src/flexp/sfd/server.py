import random
import os.path
import typing

import torch.nn
from tqdm import tqdm

from ..commons.functional import average_responses, cosine_annealing
from ..commons.evaluate import evaluate_metric
from .client import SFDClient
from .stat_agg import compute_global_stats
from .nn import make_rf_model_from_seed


class SFDServer:
    def __init__(
        self,
        *,
        clients: list[SFDClient],
        encoder: torch.nn.Module,
        classifier: torch.nn.Linear,
        projector: torch.nn.Module,
        test_loader: typing.Iterable[tuple[torch.Tensor, torch.Tensor]],
        class_num: int,
        feature_dim: int,
        global_rounds: int,
        participate_ratio: float = 1.0,
        scl_weight_start: float,
        scl_weight_end: float = 0.0,
        device: 'torch.device | str',
        result_folder: str,
    ) -> None:
        self.clients = clients.copy()
        self.encoder = encoder
        self.classifier = classifier
        self.projector = projector
        self.test_loader = test_loader
        self.class_num = class_num
        self.feature_dim = feature_dim
        self.current_round = 0
        self.global_rounds = global_rounds
        self.participate_ratio = participate_ratio
        self.device = device
        self.scl_weight_start = scl_weight_start
        self.scl_weight_end = scl_weight_end
        self.result_folder = result_folder

    def fit(self):
        assert len(self.clients) > 0, 'No clients to train.'
        for self.current_round in range(self.current_round, self.global_rounds):
            selected_clients = self.select_clients()
            scl_weight = cosine_annealing(
                init_value=self.scl_weight_start,
                total_rounds=self.global_rounds,
                current_round=self.current_round,
                last_value=self.scl_weight_end
            )
            responses: list[dict[str, typing.Any]] = []
            for client in tqdm(selected_clients, desc=f'round {self.current_round}', leave=False):
                torch.cuda.empty_cache()
                response = client.fit(
                    encoder_dict=self.encoder.state_dict(),
                    classifier_dict=self.classifier.state_dict(),
                    projector_dict=self.projector.state_dict(),
                    scl_weight=scl_weight
                )
                torch.cuda.empty_cache()
                responses.append(response)
            client_weights = [len(client.train_loader) for client in selected_clients]
            response = average_responses(responses, client_weights)
            self.encoder.load_state_dict(response['encoder'])
            self.classifier.load_state_dict(response['classifier'])
            self.projector.load_state_dict(response['projector'])
            metrics = self.test()
            print(
                f'round {self.current_round}'
                f', acc: {metrics["acc"]:.4g}'
                f', loss: {response["train_loss"]:.4g}'
                f', cls_loss: {response["cls_loss"]:.4g}'
                f', scl_loss: {response["scl_loss"]:.4g}'
            )
            checkpoint = self.make_checkpoint()
            torch.save(checkpoint, os.path.join(self.result_folder, f'latest.pt'))

    def test(self) -> dict[str, float]:
        acc = evaluate_metric(
            model=torch.nn.Sequential(self.encoder, self.classifier),
            test_loader=self.test_loader,
            device=self.device
        )['acc']
        return {
            'acc': acc
        }

    def select_clients(self):
        return (
            self.clients if self.participate_ratio == 1.0
            else random.sample(self.clients, int(len(self.clients) * self.participate_ratio))
        )

    def make_checkpoint(self) -> dict[str, typing.Any]:
        checkpoint = {
            'current_round': self.current_round,
            'encoder': self.encoder.state_dict(),
            'classifier': self.classifier.state_dict(),
            'projector': self.projector.state_dict()
        }
        return checkpoint

    def load_checkpoint(self, checkpoint: dict[str, typing.Any]):
        self.current_round = checkpoint['current_round'] + 1
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.classifier.load_state_dict(checkpoint['classifier'])
        self.projector.load_state_dict(checkpoint['projector'])

    def get_global_statistics(self, rf_dim: int, rbf_gamma: float):
        rf_seed = int(torch.randint(0, 2**32 - 1, (1,)).item())
        responses: list[dict[str, typing.Any]] = []
        for client in tqdm(self.clients, desc='calc global stats from clients', leave=False):
            response = client.get_local_statistics(
                encoder_dict=self.encoder.state_dict(),
                rf_seed=rf_seed,
                rf_dim=rf_dim,
                rbf_gamma=rbf_gamma
            )
            responses.append(response)
        global_stats = compute_global_stats(
            client_responses=responses,
            class_num=self.class_num,
            feature_dim=self.feature_dim,
            rf_dim=rf_dim
        )
        rf_model = make_rf_model_from_seed(
            seed=rf_seed,
            feature_dim=self.feature_dim,
            rf_dim=rf_dim,
            rbf_gamma=rbf_gamma,
            rf_type='orf',
            device=self.device,
        ).cpu()
        torch.cuda.empty_cache()
        if 'class_outers' in global_stats:
            del global_stats['class_outers']
        return {
            'global_stats': global_stats,
            'rf_model': rf_model,
            'rf_args': {
                'rf_dim': rf_dim,
                'rbf_gamma': rbf_gamma,
                'rf_type': 'orf',
            }
        }

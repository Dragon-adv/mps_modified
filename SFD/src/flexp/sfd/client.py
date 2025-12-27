import typing

import torch
import torch.nn
import torch.nn.functional
import torch.optim
from tqdm import tqdm

from fedbox.typing import SizedIterable
from fedbox.utils.training import AverageMeter
from .stat_agg import compute_local_stats
from .nn import make_rf_model_from_seed
import global_config


class SFDClient:
    def __init__(
        self,
        *,
        id: int,
        encoder: torch.nn.Module,
        classifier: torch.nn.Linear,  # classifier
        projector: torch.nn.Module,  # projector for contrastive learning
        train_set,
        train_loader: SizedIterable[tuple[torch.Tensor, torch.Tensor]],
        class_num: int,
        feature_dim: int,
        # --- config ---
        local_epochs: int,
        lr: float,
        momentum: float,
        weight_decay: float,
        a_ce_gamma: float,
        beta_pi: float,
        scl_temperature: float = 0.07,
        device: 'torch.device | str'
    ):
        self.id = id
        self.encoder = encoder
        self.classifier = classifier
        self.projector = projector
        self.class_num = class_num
        self.feature_dim = feature_dim
        self.train_set = train_set
        self.train_loader = train_loader
        self.local_epochs = local_epochs
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.a_ce_gamma = a_ce_gamma
        self.scl_temperature = scl_temperature
        self.device = device
        real_sample_per_class = torch.concat([y for _, y in self.train_loader]).bincount(minlength=class_num)
        if beta_pi is not None:
            self.pi_sample_per_class = make_pi_sample_per_class(real_sample_per_class, beta_pi)
        else:
            self.pi_sample_per_class = typing.cast(typing.Any, None)

    def fit(
        self,
        encoder_dict: dict[str, typing.Any],
        classifier_dict: dict[str, typing.Any],
        projector_dict: dict[str, typing.Any],
        scl_weight: float,
    ) -> dict[str, typing.Any]:
        self.encoder.load_state_dict(encoder_dict)
        self.classifier.load_state_dict(classifier_dict)
        self.projector.load_state_dict(projector_dict)
        optimizer = self.make_optimizer()
        for module in [self.encoder, self.classifier, self.projector]:
            module.to(self.device).train()
        self.pi_sample_per_class = self.pi_sample_per_class.to(self.device)
        meter = AverageMeter()
        for epoch in tqdm(range(self.local_epochs), desc=f'client {self.id}', leave=False):
            for x, y in tqdm(self.train_loader, desc=f'epoch {epoch}', leave=False):
                batch_size = x.shape[0]
                if batch_size == 1:
                    continue
                x, y = x.to(self.device), y.to(self.device)
                feature = self.encoder(x)
                logit = self.classifier(feature)
                cls_loss = logit_adjustment_ce(
                    logit=logit,
                    target=y,
                    sample_per_class=self.pi_sample_per_class,
                    gamma=self.a_ce_gamma
                )
                assert torch.isfinite(cls_loss).all(), f'cls_loss is not finite: {cls_loss}'
                proj_feature = self.projector(feature)
                contrastive_loss = a_scl_loss(
                    z=proj_feature,
                    y=y,
                    temperature=self.scl_temperature,
                    sample_per_class=self.pi_sample_per_class
                )
                assert torch.isfinite(contrastive_loss).all(), f'contrastive_loss is not finite: {contrastive_loss}'
                loss = cls_loss + scl_weight * contrastive_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                meter.add(
                    train_loss=loss.item(),
                    cls_loss=cls_loss.item(),
                    scl_loss=contrastive_loss.item(),
                )
        for module in [self.encoder, self.classifier, self.projector]:
            module.cpu()
        return {
            'encoder': self.encoder.state_dict(),
            'classifier': self.classifier.state_dict(),
            'projector': self.projector.state_dict(),
            **meter
        }

    def make_optimizer(self):
        return torch.optim.SGD(
            [
                {'params': self.encoder.parameters()},
                {'params': self.classifier.parameters()},
                {'params': self.projector.parameters()},
            ],
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )

    def get_local_statistics(
        self,
        encoder_dict: dict[str, typing.Any],
        rf_seed: int,
        rf_dim: int,
        rbf_gamma: float = 0.01
    ):
        self.encoder.load_state_dict(encoder_dict)
        rf_model = make_rf_model_from_seed(
            seed=rf_seed,
            feature_dim=self.feature_dim,
            rf_dim=rf_dim,
            rbf_gamma=rbf_gamma,
            rf_type='orf',
            device=self.device,
        )
        local_stats = compute_local_stats(
            encoder=self.encoder,
            rf_model=rf_model,
            class_num=self.class_num,
            feature_dim=self.feature_dim,
            device=self.device,
            train_set=self.train_set,
            batch_size=global_config.INFERENCE_BATCH_SIZE,
            num_workers=global_config.NUM_WORKERS
        )
        return local_stats


def make_pi_sample_per_class(real_sample_per_class: torch.Tensor, beta_pi: float) -> torch.Tensor:
    class_size_min = real_sample_per_class[real_sample_per_class > 0].min()
    pi_sample_per_class = real_sample_per_class.clone()
    pi_sample_per_class[pi_sample_per_class == 0] = class_size_min * beta_pi
    return pi_sample_per_class


def logit_adjustment_ce(
    logit: torch.Tensor,
    target: torch.Tensor,
    sample_per_class: torch.Tensor,
    gamma: float,
    reduction: str = 'mean'
) -> torch.Tensor:
    sample_per_class = (sample_per_class
        .type_as(logit)
        .unsqueeze(dim=0)
        .expand(logit.shape[0], -1))
    logit = logit + gamma * sample_per_class.log()
    loss = torch.nn.functional.cross_entropy(logit, target, reduction=reduction)
    return loss


def a_scl_loss(
    z: torch.Tensor,
    y: torch.Tensor,
    temperature: float,
    sample_per_class: torch.Tensor,
    gamma: float = 1.0
) -> torch.Tensor:
    """
    A modified supervised contrastive learning loss.
    
    Args:
        z: shape(batch_size, feature_dim)，feature vector of the sample (unnormalized)
        y: shape(batch_size,)，label corresponding to the sample
        temperature: temperature coefficient
        sample_per_class: shape(batch_size,)，number of samples in the class to which each sample belongs
    """
    z = torch.nn.functional.normalize(z, p=2, dim=1)  # Normalize feature vectors

    # 1. Calculate similarity matrix (z_i · z_j / temperature)
    similarity_matrix = (z @ z.T) / temperature

    # 2. Get mask matrix to identify positive sample pairs
    labels_equal = y.unsqueeze(0) == y.unsqueeze(1)  # True where labels are equal
    mask = labels_equal.float()  # Convert to float type for subsequent calculations

    # 3. Calculate logit adjustment
    sample_class_counts = sample_per_class[y]  # Get the number of samples in the class to which each sample belongs
    log_class_counts = torch.log(sample_class_counts.float())  # Calculate log(n_i)
    logit_adjustment_matrix = gamma * log_class_counts.unsqueeze(0)  # Expand to (1, batch_size)
    
    # 4. Generate negative sample mask to ensure adjustment term only applies to negative samples
    negative_mask = 1 - labels_equal.float()  # 1 where labels are not equal, i.e., negative sample pairs

    # 5. Apply logit adjustment to the similarity matrix of negative samples
    adjusted_similarity_matrix = similarity_matrix + logit_adjustment_matrix * negative_mask

    # Exclude self-comparison, generate a mask matrix with 0 on the diagonal
    logits_max = torch.max(adjusted_similarity_matrix, dim=1, keepdim=True).values  # Subtract the maximum value to prevent numerical instability
    logits = adjusted_similarity_matrix - logits_max.detach()  # Subtract the maximum value to prevent numerical overflow

    # Similarity scores for same-class sample pairs (logits * mask), excluding self-similarity (diagonal)
    exp_logits = torch.exp(logits) * (1 - torch.eye(z.shape[0], device=z.device))
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True))

    # 6. Calculate contrastive loss for each sample
    mean_log_prob_pos = (mask * log_prob).sum(dim=1) / mask.sum(dim=1)  # Calculate loss only for positive sample pairs

    # 7. Final supervised contrastive learning loss
    loss = -mean_log_prob_pos.mean()  # Calculate the average loss, take the negative
    return loss

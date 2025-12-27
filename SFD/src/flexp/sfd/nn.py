import random
import math

import numpy as np
import torch
import torch.backends.cudnn


class RFF(torch.nn.Module):
    def __init__(self, d: int, D: int, gamma: float, device: 'torch.device | str', rf_type: str):
        super(RFF, self).__init__()
        assert D % 2 == 0, "D must be an even number"
        self.d = d
        self.D = D
        self.gamma = gamma
        """
        gamma = 1 / (2 * sigma^2)

        sigma = 1 / sqrt(2 * gamma)
        """
        sigma = 1 / math.sqrt(2 * gamma)
        # Initialize weights with standard normal distribution and then scale by 1/sigma
        if rf_type == 'iid':
            w = torch.randn(D // 2, d, device=device) * (1 / sigma)
        elif rf_type == 'orf':
            w = make_orf_matrix(d, D // 2, std=1 / sigma, device=device).T
        else:
            raise NotImplementedError(f'Random Features type "{rf_type}" is not implemented.')
        self.register_buffer('w', w)
        self.w: torch.Tensor

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        :param X: Input tensor of shape (batch_size, d)
        :return: Transformed tensor of shape (batch_size, D)
        """
        Xw = torch.matmul(X, self.w.T)  # Shape: (batch_size, D//2)
        # Compute cos and sin features
        Z_cos = torch.cos(Xw)
        Z_sin = torch.sin(Xw)
        # Concatenate cos and sin features along the last dimension
        Z = torch.cat([Z_cos, Z_sin], dim=-1)
        # Apply normalization factor sqrt(2 / D)
        Z = Z * math.sqrt(2 / self.D)
        return Z


class MeanCovAligner(torch.nn.Module):
    def __init__(
        self,
        target_mean: torch.Tensor,
        target_cov: torch.Tensor,
        target_cov_eps: float,
    ):
        """
        :param target_mean: shape (d,)
        :param target_cov: shape (d, d)
        """
        super().__init__()
        self.register_buffer('target_mean', target_mean)
        self.target_mean: torch.Tensor
        self.register_buffer('target_cov_eps', torch.tensor(target_cov_eps))
        self.target_cov_eps: torch.Tensor
        # add epsilon to the diagonal of the target covariance matrix
        d = target_cov.shape[0]
        target_cov = (
            target_cov.to(dtype=torch.float64, device='cpu')
            + target_cov_eps * torch.eye(d, dtype=torch.float64)
        )
        try:
            L2 = torch.linalg.cholesky_ex(target_cov, check_errors=True)[0]
        except RuntimeError as e:
            raise RuntimeError(
                f'Cholesky decomposition failed for target_cov with eps={target_cov_eps}. '
                f'Please check if the input is valid or try a larger target_cov_eps.'
            ) from e
        assert torch.isfinite(L2).all(), \
            f'Non-finite values in L2: {L2}, check if the input is valid or try a larger target_cov_eps.'
        self.register_buffer('L2', L2)
        self.L2: torch.Tensor
        assert self.L2.dtype == torch.float64

    def forward(
        self,
        data: torch.Tensor,
        *,
        decompose_dtype: torch.dtype = torch.float64,
        input_cov_eps: float,
    ) -> torch.Tensor:
        """
        :param data: shape (n, d)
        """
        d = data.shape[1]
        mean1 = data.mean(dim=0)
        cov1 = data.T.cov()  # shape (d, d)
        cov1 = cov1.to(decompose_dtype) + input_cov_eps * torch.eye(d, dtype=cov1.dtype, device=cov1.device)
        try:
            L1 = torch.linalg.cholesky_ex(cov1, check_errors=True)[0]
        except RuntimeError as e:
            raise RuntimeError(
                f'Cholesky decomposition failed for input covariance with eps={input_cov_eps}. '
                f'Please check if the input is valid or try a larger input_cov_eps.'
            ) from e
        assert torch.isfinite(L1).all(), \
            f'Non-finite values in L1: {L1}, check if the input is valid or try a larger input_cov_eps.'
        assert L1.dtype == decompose_dtype
        # A = L2 @ L1^-1
        # L1_inv = torch.linalg.solve(L1, torch.eye(d, dtype=L1.dtype, device=L1.device))
        # A = (self.L2.to(decompose_dtype) @ L1_inv).to(data.dtype)
        A = torch.linalg.solve_triangular(L1, self.L2.to(L1.dtype), upper=False, left=False).to(data.dtype)
        assert torch.isfinite(A).all(), f'Non-finite values in A: {A}'
        aligned_data = (data - mean1) @ A.T + self.target_mean
        return aligned_data


def make_orf_matrix(dim_in: int, dim_out: int, std: float, device):
    """
    Generates random matrix of orthogonal random features using PyTorch (optimized for batch computation).

    Args:
        dim_in  (int)  : Input dimension of the random matrix.
        dim_out (int)  : Output dimension of the random matrix.
        std     (float): Standard deviation of the random matrix.

    Returns:
        (torch.Tensor): Random matrix with shape (dim_out, dim_in).
    """
    # Number of batches required to cover the output dimensions
    num_batches = dim_out // dim_in + 1
    
    # Step 1: Batch generate random normal matrices
    # Shape: (num_batches, dim_in, dim_in)
    rand_matrices = torch.randn((num_batches, dim_in, dim_in), device=device)
    
    # Step 2: Perform batch QR decomposition
    # QR decomposition along the last two dimensions
    Q_matrices = torch.linalg.qr(rand_matrices).Q  # Shape: (num_batches, dim_in, dim_in)

    # Step 3: Batch sample from chi distribution and compute sqrt(chi)
    chi2_dist = torch.distributions.Chi2(df=torch.tensor(dim_in, device=device))
    # Batch sample from Chi-squared distribution and take sqrt to get Chi distribution
    s_batch = chi2_dist.sample(torch.Size([num_batches, dim_in])).sqrt()  # Shape: (num_batches, dim_in)

    # Step 4: Scale each orthogonal matrix by the sampled Chi values
    # We first create diagonal matrices from the chi samples
    diag_s_batch = torch.stack([torch.diag(s) for s in s_batch])  # Shape: (num_batches, dim_in, dim_in)
    
    # Step 5: Scale the orthogonal matrices
    V_matrices = std * torch.bmm(diag_s_batch, Q_matrices)  # Shape: (num_batches, dim_in, dim_in)
    
    # Step 6: Reshape V_matrices to concatenate all into a single matrix
    # Reshape to (dim_in, num_batches * dim_in) and slice the first dim_out columns
    W_full = V_matrices.transpose(0, 1).reshape(dim_in, -1)  # Shape: (dim_in, num_batches * dim_in)

    # Step 7: Trim the matrix to match the required shape (dim_out, dim_in)
    return W_full[:, :dim_out]


def backup_rng_states():
    """Backup the current state of all random number generators"""
    backup = {
        'python_random': random.getstate(),
        'numpy': np.random.get_state(),
        'torch_cpu': torch.get_rng_state(),
        'cuda': [],
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
        'cudnn_benchmark': torch.backends.cudnn.benchmark
    }
    if torch.cuda.is_available():
        for device in range(torch.cuda.device_count()):
            backup['cuda'].append(torch.cuda.get_rng_state(device))
    return backup


def set_deterministic_mode(seed: int):
    """Set all components to deterministic mode and initialize the seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def restore_rng_states(backup: dict):
    """Restore the state of all random number generators from a backup"""
    random.setstate(backup['python_random'])
    np.random.set_state(backup['numpy'])
    torch.set_rng_state(backup['torch_cpu'])
    if torch.cuda.is_available():
        for device, state in enumerate(backup['cuda']):
            torch.cuda.set_rng_state(state, device)
    torch.backends.cudnn.deterministic = backup['cudnn_deterministic']
    torch.backends.cudnn.benchmark = backup['cudnn_benchmark']


def make_rf_model_from_seed(
    seed: int,
    feature_dim: int,
    rf_dim: int,
    rbf_gamma: float,
    rf_type: str,
    device: 'torch.device | str'
):
    backup = backup_rng_states()
    set_deterministic_mode(seed)
    rf_model = RFF(
        d=feature_dim,
        D=rf_dim,
        gamma=rbf_gamma,
        device=device,
        rf_type=rf_type
    )
    restore_rng_states(backup)
    return rf_model


def assert_rf_models_all_equal(rf_models: list[RFF]):
    """
    Assert that all RFF models in the list have the same parameters.
    """
    if not rf_models:
        return
    first_model = rf_models[0]
    for model in rf_models[1:]:
        assert first_model.d == model.d, f'Input dimension mismatch: {first_model.d} != {model.d}'
        assert first_model.D == model.D, f'Output dimension mismatch: {first_model.D} != {model.D}'
        assert first_model.gamma == model.gamma, f'Gamma mismatch: {first_model.gamma} != {model.gamma}'
        assert torch.equal(first_model.w, model.w), 'Weight matrices are not equal'

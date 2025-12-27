from typing import Mapping, Sequence, Any

import numpy as np
import torch
from fedbox.utils.functional import model_average, weighted_average as tensor_average


def average_responses(responses: Sequence[dict[str, Any]], client_weights: Sequence[float]) -> dict[str, Any]:
    averaged = {}
    for key, value in responses[0].items():
        if isinstance(value, Mapping):  # state dict
            averaged[key] = model_average([response[key] for response in responses], client_weights)
        elif isinstance(value, torch.Tensor):  # pytorch tensor
            averaged[key] = tensor_average([response[key] for response in responses], weights=client_weights)
        elif isinstance(value, (int, float)):  # scalar
            averaged[key] = np.average([response[key] for response in responses], weights=client_weights).item()
        elif isinstance(value, np.ndarray):  # numpy array
            averaged[key] = np.average([response[key] for response in responses], weights=client_weights)
        else:
            raise ValueError(f'Unsupported type: {type(value)}')
    return averaged


def cosine_annealing(
    init_value: float,
    total_rounds: int,
    current_round: int,
    last_value: float = 0.0
) -> float:
    """
    Computes a value that decreases from `init_value` to `last_value` using cosine annealing.

    Args:
        init_value (float): The initial value at the start of the rounds.
        total_rounds (int): The total number of rounds.
        current_round (int): The current round number.
        last_value (float, optional): The value at the end of the rounds. Defaults to 0.0.

    Returns:
        float: The computed value for the current round.
    """
    assert 0 <= current_round < total_rounds, 'current_round must be in the range [0, total_rounds)'
    return last_value + 0.5 * (init_value - last_value) * (1 + np.cos(current_round / total_rounds * np.pi))

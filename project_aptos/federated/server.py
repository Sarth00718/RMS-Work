"""
Federated Learning Server.

Implements FedAvg aggregation with weighted averaging
based on client sample counts.
"""
import copy
import torch
import torch.nn as nn
from typing import Dict, List, Tuple

class FLServer:
    """
    Federated Learning server that aggregates weights from multiple clients.
    """
    def __init__(self, model: nn.Module):
        self.global_model = model
        self.global_weights = {k: v.cpu() for k, v in model.state_dict().items()}
        self.median_post_acc = 0.0

    def get_global_weights(self) -> Dict[str, torch.Tensor]:
        return copy.deepcopy(self.global_weights)

    def get_median_post_acc(self) -> float:
        return self.median_post_acc

    def aggregate_weights(self, client_updates: List[Tuple[int, Dict[str, torch.Tensor], float]]):
        """
        FedAvg: Aggregates client weights.
        client_updates is a list of tuples: (num_samples, weights_dict, post_acc)
        """
        if not client_updates:
            return

        total_samples = sum(num_samples for num_samples, _, _ in client_updates)

        # Initialize an empty dictionary for the new global weights (use float32 to prevent Long casting issues)
        new_global_weights = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in self.global_weights.items()}

        post_accs = []
        for num_samples, client_weights, post_acc in client_updates:
            post_accs.append(post_acc)
            weight_fraction = num_samples / total_samples
            for k in new_global_weights.keys():
                new_global_weights[k] += client_weights[k].to(torch.float32) * weight_fraction

        # Cast back to original data types
        for k in self.global_weights.keys():
            self.global_weights[k] = new_global_weights[k].to(self.global_weights[k].dtype)

        self.global_model.load_state_dict(self.global_weights)

        # Update median post accuracy for next round's checkpoint 1
        if post_accs:
            import numpy as np
            self.median_post_acc = float(np.median(post_accs))

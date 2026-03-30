import torch
import torch.nn as nn

# Input dimension matches the 15 real clinical features in data_generator.py
INPUT_DIM = 15


class DiseaseRiskModel(nn.Module):
    """
    Neural network for cardiovascular disease risk prediction.

    NOTE: BatchNorm is intentionally excluded. In federated learning,
    weight serialization uses named_parameters() which skips BatchNorm
    buffers (running_mean / running_var). Using BatchNorm causes the
    model to output ~0 for every input in eval() mode because the
    running statistics are never synchronized across clients → fixed by
    using LayerNorm-free architecture with simple Dropout regularisation.
    """
    def __init__(self, input_dim: int = INPUT_DIM, hidden_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            # Layer 1
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Layer 2
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.25),

            # Layer 3
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),

            # Output – risk probability
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def get_model(input_dim: int = INPUT_DIM) -> DiseaseRiskModel:
    return DiseaseRiskModel(input_dim=input_dim)

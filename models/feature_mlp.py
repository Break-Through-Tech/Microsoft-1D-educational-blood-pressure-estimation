"""Feature-based SBP/DBP regression for issue #29.

One row of PPG features goes in; two continuous predictions come out.
This model does not load waveforms, clean signals, or calculate ABP labels.
"""

from torch import nn


class FeatureMLP(nn.Module):
    """Input: (batch, n_features). Output: (batch, 2), ordered SBP then DBP."""

    def __init__(self, n_features: int, dropout: float = 0.1, hidden_sizes=(32, 16)):
        super().__init__()
        if not isinstance(n_features, int) or isinstance(n_features, bool) or n_features < 1:
            raise ValueError("n_features must be a positive integer")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if not isinstance(hidden_sizes, (tuple, list)) or not hidden_sizes or any(
            not isinstance(n, int) or isinstance(n, bool) or n < 1 for n in hidden_sizes
        ):
            raise ValueError("hidden_sizes must contain positive integers")
        layers, previous = [], n_features
        for i, width in enumerate(hidden_sizes):
            layers.extend([nn.Linear(previous, width), nn.ReLU()])
            if i < len(hidden_sizes) - 1:
                layers.append(nn.Dropout(dropout))
            previous = width
        # Default layer positions remain identical to historical checkpoints.
        layers.append(nn.Linear(previous, 2))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

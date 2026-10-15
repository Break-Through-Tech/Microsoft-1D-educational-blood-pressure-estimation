"""Small 1D convolutional network that estimates SBP and DBP from one raw PPG window.

Input shape:  (batch, 1, 625)  - one channel, 625 samples = 5 seconds at 125 Hz.
Output shape: (batch, 2)       - [SBP, DBP], in whatever units the targets were scaled to.
"""

from torch import nn


class PPGCNN(nn.Module):
    """Three convolution blocks, a global average, and one linear output layer."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 16 learned filters, each 7 samples (~56 ms) wide, slide along the waveform
            # and respond to short local shapes such as a steep upstroke. padding keeps the length.
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            # ReLU keeps positive responses and zeroes negative ones, letting the network model
            # non-linear relationships.
            nn.ReLU(),
            # Max pooling halves the length (625 -> 312) by keeping the strongest response in each
            # pair of positions, so later layers see a wider stretch of signal at lower cost.
            nn.MaxPool1d(2),
            # Block 2: combines block-1 patterns into larger shapes (312 -> 156 after pooling).
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            # Block 3: larger patterns again, roughly pulse-sized (156 -> 78 after pooling).
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            # Average each of the 64 filter responses over the whole window (78 -> 1). The result
            # describes which patterns are present, regardless of where in the 5 seconds they occur.
            nn.AdaptiveAvgPool1d(1),
        )
        # Linear layer: a weighted sum of the 64 pattern scores gives the two outputs, SBP and DBP.
        self.head = nn.Linear(64, 2)

    def forward(self, x):
        x = self.features(x)  # (batch, 64, 1)
        x = x.flatten(1)      # (batch, 64)
        return self.head(x)   # (batch, 2)

"""Small 1D convolutional network that estimates SBP and DBP from one raw PPG window.

Input shape:  (batch, 1, 625)  - one channel, 625 samples = 5 seconds at 125 Hz.
Output shape: (batch, 2)       - [SBP, DBP], in whatever units the targets were scaled to.
"""

from torch import nn


class PPGCNN(nn.Module):
    """Three convolution blocks, a global average, and one linear output layer."""

    def __init__(self, input_channels=1, channels=(16, 32, 64), dilations=(1, 1, 1)):
        super().__init__()
        if input_channels not in (1, 3) or len(channels) != 3 or any(int(c) != c or c < 1 for c in channels):
            raise ValueError("Expected one or three input channels and three positive channel widths")
        a, b, c = channels
        if len(dilations) != 3 or any(isinstance(d,bool) or int(d) != d or d < 1 for d in dilations):
            raise ValueError("Expected three positive integer dilations")
        d1, d2, d3 = dilations
        self.features = nn.Sequential(
            # Block 1: 16 learned filters, each 7 samples (~56 ms) wide, slide along the waveform
            # and respond to short local shapes such as a steep upstroke. padding keeps the length.
            nn.Conv1d(input_channels, a, kernel_size=7, padding=3*d1, dilation=d1),
            # ReLU keeps positive responses and zeroes negative ones, letting the network model
            # non-linear relationships.
            nn.ReLU(),
            # Max pooling halves the length (625 -> 312) by keeping the strongest response in each
            # pair of positions, so later layers see a wider stretch of signal at lower cost.
            nn.MaxPool1d(2),
            # Block 2: combines block-1 patterns into larger shapes (312 -> 156 after pooling).
            nn.Conv1d(a, b, kernel_size=5, padding=2*d2, dilation=d2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            # Block 3: larger patterns again, roughly pulse-sized (156 -> 78 after pooling).
            nn.Conv1d(b, c, kernel_size=5, padding=2*d3, dilation=d3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            # Average each of the 64 filter responses over the whole window (78 -> 1). The result
            # describes which patterns are present, regardless of where in the 5 seconds they occur.
            nn.AdaptiveAvgPool1d(1),
        )
        # Linear layer: a weighted sum of the 64 pattern scores gives the two outputs, SBP and DBP.
        self.head = nn.Linear(c, 2)

    def forward(self, x):
        x = self.features(x)  # (batch, 64, 1)
        x = x.flatten(1)      # (batch, 64)
        return self.head(x)   # (batch, 2)


class ResidualBlock1D(nn.Module):
    """Two convolution/BN layers with a dimension-matching shortcut."""

    def __init__(self, incoming, outgoing, stride=1, kernel_size=5):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(incoming, outgoing, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm1d(outgoing), nn.ReLU(),
            nn.Conv1d(outgoing, outgoing, kernel_size, 1, kernel_size//2, bias=False),
            nn.BatchNorm1d(outgoing))
        self.shortcut = nn.Identity() if incoming == outgoing and stride == 1 else nn.Sequential(
            nn.Conv1d(incoming, outgoing, 1, stride, bias=False), nn.BatchNorm1d(outgoing))
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(self.body(x) + self.shortcut(x))


class ResidualPPGCNN(nn.Module):
    """Compact post-activation residual comparator, not a standard ResNet-18."""

    def __init__(self, input_channels=3, channels=(16, 32, 64), kernel_size=5,
                 auxiliary_classification=False):
        super().__init__()
        if input_channels != 3 or len(channels)!=3 or any(isinstance(c,bool) or int(c)!=c or c<1 for c in channels):
            raise ValueError('Residual model requires three inputs and three positive integer widths')
        if kernel_size not in (5,7):raise ValueError('Residual kernels must be five or seven')
        a,b,c=channels
        self.features = nn.Sequential(
            nn.Conv1d(3, a, 7, 2, 3, bias=False), nn.BatchNorm1d(a), nn.ReLU(),
            ResidualBlock1D(a, a, kernel_size=kernel_size), ResidualBlock1D(a, b, 2, kernel_size),
            ResidualBlock1D(b, c, 2, kernel_size))
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(c, 2)
        # Construct auxiliary heads last: shared/regression initialization stays matched.
        if auxiliary_classification:
            self.sbp_classifier=nn.Linear(c,5)
            self.dbp_classifier=nn.Linear(c,5)

    def forward(self, x):
        return self.head(self.pool(self.features(x)).flatten(1))

    def forward_tasks(self,x):
        if not hasattr(self,'sbp_classifier'):raise ValueError('No auxiliary classification heads')
        shared=self.pool(self.features(x)).flatten(1)
        return self.head(shared),self.sbp_classifier(shared),self.dbp_classifier(shared)


def build_wave_model(parameters):
    """Absent architecture retains historical checkpoint construction exactly."""
    architecture = parameters.get('architecture', 'short_cnn')
    inputs = 3 if parameters['representation'] == 'derivatives' else 1
    if architecture == 'short_cnn':
        return PPGCNN(inputs, parameters['channels'], parameters.get('dilations', (1, 1, 1)))
    if architecture == 'residual_v1':
        return ResidualPPGCNN(inputs, parameters['channels'],parameters.get('kernel_size',5),
            parameters.get('auxiliary_classification',False))
    raise ValueError('Unknown waveform architecture: ' + str(architecture))

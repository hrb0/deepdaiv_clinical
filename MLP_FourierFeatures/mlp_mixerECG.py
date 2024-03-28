import torch
import torch.nn as nn


# Redefine the FeedForward module to ensure the input dimension matches the first Linear layer's expectation
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),  # Ensure the input dimension 'dim' matches this layer's expectation
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )


    def forward(self, x):

        return self.net(x)


# No changes needed for ECG
# MixerBlock as it properly configures FeedForward
class ECGMixerBlock(nn.Module):

    def __init__(self, dim, num_patch, token_dim, channel_dim, dropout=0.):

        super().__init__()

        # Token mixing
        self.token_mix = nn.Sequential(
            nn.LayerNorm(dim),
            FeedForward(dim, token_dim, dropout)  # Ensure the first Linear layer of FeedForward matches 'dim'
        )

        # Channel mixing
        self.channel_mix = nn.Sequential(
            nn.LayerNorm(dim),
            FeedForward(dim, channel_dim, dropout)
        )


    def forward(self, x):

        x = x + self.token_mix(x)

        x = x + self.channel_mix(x)

        return x


# The MLPMixerForECG class remains the same
class MLPMixerForECG(nn.Module):

    def __init__(self, channels, dim, num_classes, seq_len, depth, token_dim, channel_dim):

        super().__init__()

        self.seq_len = seq_len

        self.to_patch_embedding = nn.Sequential(
            nn.Conv1d(channels, dim, kernel_size=1),
        )

        self.mixer_blocks = nn.ModuleList([])

        for _ in range(depth):
            self.mixer_blocks.append(ECGMixerBlock(dim, self.seq_len, token_dim, channel_dim))

        self.layer_norm = nn.LayerNorm(dim)

        self.mlp_head = nn.Sequential(
            nn.Linear(dim, num_classes)
        )


    def forward(self, x):

        x = self.to_patch_embedding(x)

        x = x.transpose(1, 2)  # Transpose to have [batch, seq_len, dim]

        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)

        x = self.layer_norm(x)
        
        x = x.mean(dim=1)  # Global average pooling equivalent

        return self.mlp_head(x)


# Commenting out the example usage to prevent execution in the PCI
if __name__ == "__main__":

    ecg_signal = torch.randn([1, 1, 500])  # Example ECG signal batch with size [batch, channels, length]

    model = MLPMixerForECG(channels=1, seq_len=500, num_classes=2, dim=128, depth=6, token_dim=64, channel_dim=256)

    output = model(ecg_signal)

    print(output.shape)  # Expected output shape: [1, num_classes]


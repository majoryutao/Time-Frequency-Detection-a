import torch
import torch.nn as nn


class TFCM(nn.Module):
    """Time-Frequency Continuity Module.

    layout:
        - "FT": x is [B, C, F, T], i.e. H=freq, W=time
        - "TF": x is [B, C, T, F], i.e. H=time, W=freq

    Default uses "FT", which matches the common spectrogram image layout:
    vertical axis = frequency, horizontal axis = time.
    """

    def __init__(self, c1, c2, kt=9, kf=9, reduction=16, shortcut=True, layout="FT"):
        super().__init__()
        if layout not in ("FT", "TF"):
            raise ValueError(f"Unsupported layout: {layout}. Use 'FT' or 'TF'.")

        self.shortcut = shortcut
        self.layout = layout
        self.proj = Conv(c1, c2, 1, 1) if c1 != c2 else nn.Identity()

        # Make axis semantics explicit instead of relying on ambiguous names.
        if layout == "FT":
            # x: [B, C, F, T] => H=freq, W=time
            time_kernel = (1, kt)
            freq_kernel = (kf, 1)
        else:
            # x: [B, C, T, F] => H=time, W=freq
            time_kernel = (kt, 1)
            freq_kernel = (1, kf)

        self.time_branch = nn.Sequential(
            nn.Conv2d(
                c2, c2,
                kernel_size=time_kernel,
                stride=1,
                padding=(time_kernel[0] // 2, time_kernel[1] // 2),
                groups=c2,
                bias=False,
            ),
            nn.BatchNorm2d(c2),
            nn.SiLU(),
        )

        self.freq_branch = nn.Sequential(
            nn.Conv2d(
                c2, c2,
                kernel_size=freq_kernel,
                stride=1,
                padding=(freq_kernel[0] // 2, freq_kernel[1] // 2),
                groups=c2,
                bias=False,
            ),
            nn.BatchNorm2d(c2),
            nn.SiLU(),
        )

        self.local = Conv(c2, c2, 3, 1)

        self.fuse = Conv(c2 * 3, c2, 1, 1)
        hidden = max(c2 // reduction, 8)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c2, hidden, 1, bias=False),
            nn.SiLU(),
            nn.Conv2d(hidden, c2, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError(f"TFCM expects 4D input [B, C, H, W], got shape {tuple(x.shape)}")

        x0 = self.proj(x)
        ft = self.time_branch(x0)
        ff = self.freq_branch(x0)
        fl = self.local(x0)

        y = self.fuse(torch.cat((ft, ff, fl), dim=1))
        y = y * self.gate(y)

        return x0 + y if self.shortcut else y


class TFCMLite(nn.Module):
    """Lightweight TFCM for higher-level features.

    layout:
        - "FT": x is [B, C, F, T]
        - "TF": x is [B, C, T, F]
    """

    def __init__(self, c1, c2, kt=7, kf=7, reduction=32, shortcut=True, layout="FT"):
        super().__init__()
        if layout not in ("FT", "TF"):
            raise ValueError(f"Unsupported layout: {layout}. Use 'FT' or 'TF'.")

        self.shortcut = shortcut
        self.layout = layout
        self.proj = Conv(c1, c2, 1, 1) if c1 != c2 else nn.Identity()

        if layout == "FT":
            time_kernel = (1, kt)
            freq_kernel = (kf, 1)
        else:
            time_kernel = (kt, 1)
            freq_kernel = (1, kf)

        self.time_branch = nn.Sequential(
            nn.Conv2d(
                c2, c2,
                kernel_size=time_kernel,
                stride=1,
                padding=(time_kernel[0] // 2, time_kernel[1] // 2),
                groups=c2,
                bias=False,
            ),
            nn.BatchNorm2d(c2),
            nn.SiLU(),
        )

        self.freq_branch = nn.Sequential(
            nn.Conv2d(
                c2, c2,
                kernel_size=freq_kernel,
                stride=1,
                padding=(freq_kernel[0] // 2, freq_kernel[1] // 2),
                groups=c2,
                bias=False,
            ),
            nn.BatchNorm2d(c2),
            nn.SiLU(),
        )

        self.fuse = Conv(c2 * 2, c2, 1, 1)
        hidden = max(c2 // reduction, 8)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c2, hidden, 1, bias=False),
            nn.SiLU(),
            nn.Conv2d(hidden, c2, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError(f"TFCMLite expects 4D input [B, C, H, W], got shape {tuple(x.shape)}")

        x0 = self.proj(x)
        ft = self.time_branch(x0)
        ff = self.freq_branch(x0)

        y = self.fuse(torch.cat((ft, ff), dim=1))
        y = y * self.gate(y)

        return x0 + y if self.shortcut else y

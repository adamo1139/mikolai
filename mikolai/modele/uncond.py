import torch
import torch.nn as nn


class UncondParams(nn.Module):
    """Uczone tokeny „bez warunku” do CFG (z training/t2l/cfg_utils.py)."""
    def __init__(self, text_dim=256, n_style=50, style_value_dim=256, init_std=0.1, uncond_key=False, style_key_dim=256):
        super().__init__()
        self.u_text = nn.Parameter(torch.randn(1, text_dim, 1) * init_std)
        self.u_ref = nn.Parameter(torch.randn(1, n_style, style_value_dim) * init_std)
        if uncond_key:
            self.u_key = nn.Parameter(torch.randn(1, n_style, style_key_dim) * init_std)

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.layers import DropPath, to_2tuple

class ConvBnGELU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, stride, padding=k//2, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()
        
    def forward(self, x): 
        return self.act(self.bn(self.conv(x)))

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, dropout_rate: float = 0.0):
        super().__init__()
        self.conv1 = ConvBnGELU(in_ch, out_ch, k=k)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.short = nn.Identity() if in_ch == out_ch else nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch)
        )
        self.act = nn.GELU()
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x):
        r = self.conv1(x)
        r = self.bn2(self.conv2(r))
        return self.dropout(self.act(r + self.short(x)))

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 1)
        self.channel_gate = nn.Sequential(
            nn.Conv2d(channels, mid, 1), 
            nn.ReLU(True), 
            nn.Conv2d(mid, channels, 1)
        )
        self.spatial_gate = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        
    def forward(self, x):
        ch_att = torch.sigmoid(
            self.channel_gate(F.adaptive_avg_pool2d(x, 1)) +
            self.channel_gate(F.adaptive_max_pool2d(x, 1))
        )
        x = x * ch_att
        sp_att = torch.sigmoid(
            self.spatial_gate(
                torch.cat([torch.mean(x, dim=1, keepdim=True),
                           torch.max(x, dim=1, keepdim=True)[0]], dim=1)
            )
        )
        x = x * sp_att
        return x

class LesionGate(nn.Module):
    def __init__(self, in_channels, feature_channels):
        super().__init__()
        self.gate_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 4), 
            nn.GELU(), 
            nn.Conv2d(in_channels // 4, feature_channels, kernel_size=1)
        )

    def forward(self, x_up, skip_features):
        gate_upsampled = F.interpolate(x_up, size=skip_features.shape[2:], mode='bilinear', align_corners=False)
        spatial_gate = self.gate_conv(gate_upsampled)
        return skip_features + (skip_features * torch.sigmoid(spatial_gate))

class EncoderStage(nn.Module):
    def __init__(self, in_ch, out_ch, conv_kernels: List[int]):
        super().__init__()
        layers = []
        cur_in = in_ch
        for k in conv_kernels:
            layers.append(ConvBnGELU(cur_in, out_ch, k=k))
            cur_in = out_ch
        self.net = nn.Sequential(*layers)
        
    def forward(self, x): 
        return self.net(x)

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=8, mlp_ratio=4., drop_path=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size=to_2tuple(self.window_size), num_heads=num_heads)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_hidden_dim), nn.GELU(), nn.Linear(mlp_hidden_dim, dim))

    def forward(self, x, attn_mask=None):
        B, C, H, W = x.shape
        shortcut = x
        x_ln = self.norm1(rearrange(x, 'b c h w -> b (h w) c'))
        x_ln_2d = rearrange(x_ln, 'b (h w) c -> b h w c', h=H, w=W)
        win_size = min(self.window_size, H, W)
        pad_l = pad_t = 0
        pad_r = (win_size - W % win_size) % win_size
        pad_b = (win_size - H % win_size) % win_size
        x_padded = F.pad(x_ln_2d, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, H_pad, W_pad, _ = x_padded.shape
        x_windows = window_partition(x_padded, win_size)
        x_windows = x_windows.view(-1, win_size * win_size, C)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_2d = window_reverse(attn_windows, win_size, H_pad, W_pad)
        if pad_r > 0 or pad_b > 0:
            attn_2d = attn_2d[:, :H, :W, :].contiguous()
        
        attn_out = rearrange(attn_2d, 'b h w c -> b c h w')
        x = shortcut + self.drop_path(attn_out)
        mlp_shortcut = x
        x_mlp_out = self.mlp(self.norm2(rearrange(x, 'b c h w -> b (h w) c')))
        x = mlp_shortcut + self.drop_path(rearrange(x_mlp_out, 'b (h w) c -> b c h w', h=H, w=W))
        return x

class SwinTransformerLayer(nn.Module):
    def __init__(self, dim, num_heads, window_size=8, mlp_ratio=4., drop_path=0.):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.block1 = SwinTransformerBlock(dim, num_heads, window_size, mlp_ratio, drop_path)
        self.block2 = SwinTransformerBlock(dim, num_heads, window_size, mlp_ratio, drop_path)
        self.attn_mask = None

    def create_mask(self, x):
        _, _, H, W = x.shape
        if self.attn_mask is None or self.attn_mask.shape[-1] != W or self.attn_mask.shape[-2] != H:
            img_mask = torch.zeros((1, H, W, 1), device=x.device)
            h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            pad_r = (self.window_size - W % self.window_size) % self.window_size
            pad_b = (self.window_size - H % self.window_size) % self.window_size
            img_mask = F.pad(img_mask, (0, 0, 0, pad_r, 0, pad_b))
            H_pad, W_pad = img_mask.shape[1], img_mask.shape[2]
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            self.attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return self.attn_mask

    def forward(self, x):
        x = self.block1(x, attn_mask=None)
        shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
        mask = self.create_mask(shifted_x)
        shifted_x = self.block2(shifted_x, attn_mask=mask)
        x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(2, 3))
        return x

class FusionBottleneck(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.crossA = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.crossB = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.normA = nn.LayerNorm(dim)
        self.normB = nn.LayerNorm(dim)

    def forward(self, tokA, tokB):
        attA2B, _ = self.crossA(tokA, tokB, tokB)
        attB2A, _ = self.crossB(tokB, tokA, tokA)
        enrichedA = self.normA(tokA + attA2B)
        enrichedB = self.normB(tokB + attB2A)
        return enrichedA, enrichedB

def window_partition(x, window_size: int):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size: int, H: int, W: int):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x
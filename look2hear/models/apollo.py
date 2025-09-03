import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from .base_model import BaseModel

class RMSNorm(nn.Module):
    def __init__(self, dimension, groups=1):
        super().__init__()
        
        self.weight = nn.Parameter(torch.ones(dimension))
        self.groups = groups
        self.eps = 1e-5

    def forward(self, input):
        # input size: (B, N, T)
        B, N, T = input.shape
        assert N % self.groups == 0

        input_float = input.reshape(B, self.groups, -1, T).float()
        input_norm = input_float * torch.rsqrt(input_float.pow(2).mean(-2, keepdim=True) + self.eps)

        return input_norm.type_as(input).reshape(B, N, T) * self.weight.reshape(1, -1, 1)
    
class RMVN(nn.Module):
    """
    Rescaled MVN.
    """
    def __init__(self, dimension, groups=1):
        super(RMVN, self).__init__()
        
        self.mean = nn.Parameter(torch.zeros(dimension))
        self.std = nn.Parameter(torch.ones(dimension))
        self.groups = groups
        self.eps = 1e-5

    def forward(self, input):
        # input size: (B, N, *)
        B, N = input.shape[:2]
        assert N % self.groups == 0
        input_reshape = input.reshape(B, self.groups, N // self.groups, -1)
        T = input_reshape.shape[-1]

        input_norm = (input_reshape - input_reshape.mean(2).unsqueeze(2)) / (input_reshape.var(2).unsqueeze(2) + self.eps).sqrt()
        input_norm = input_norm.reshape(B, N, T) * self.std.reshape(1, -1, 1) + self.mean.reshape(1, -1, 1)

        return input_norm.reshape(input.shape)
    
class WMSA1D(nn.Module):
    """
    Window-based Multi-Head Self-Attention (1D) inspired by Uformer.
    Operates on the sequence dimension (T) using non-overlapping windows.
    Complexity reduces from O(T^2) to O(T * W), where W is window size.
    """
    def __init__(self, input_size, hidden_size, num_head=8, window=8,
                 input_drop=0., attention_drop=0.):
        super().__init__()

        assert hidden_size % num_head == 0, "hidden_size must be divisible by num_head"
        self.input_size = input_size
        self.hidden_size = hidden_size // num_head
        self.num_head = num_head
        self.window_size = max(1, int(window))
        self.attention_drop = attention_drop

        self.input_norm = RMSNorm(self.input_size)
        self.input_drop = nn.Dropout(p=input_drop)
        self.qkv = nn.Conv1d(self.input_size, self.hidden_size*self.num_head*3, 1, bias=False)
        self.proj = nn.Conv1d(self.hidden_size*self.num_head, self.input_size, 1, bias=False)

        # MLP block (gated) similar to the original module for parity
        self.MLP = nn.Sequential(
            RMSNorm(self.input_size),
            nn.Conv1d(self.input_size, self.input_size*8, 1, bias=False),
            nn.SiLU()
        )
        self.MLP_output = nn.Conv1d(self.input_size*4, self.input_size, 1, bias=False)

    def _pad_to_window(self, x, ws):
        # x: (B, H, T, C)
        T = x.shape[2]
        pad_len = (ws - (T % ws)) % ws
        if pad_len > 0:
            pad = torch.zeros((*x.shape[:2], pad_len, x.shape[3]), dtype=x.dtype, device=x.device)
            x = torch.cat([x, pad], dim=2)
        return x, pad_len

    def forward(self, input):
        # input: (B, N, T)
        B, _, T = input.shape
        ws = min(self.window_size, max(1, T))

        qkv = self.qkv(self.input_drop(self.input_norm(input)))  # (B, 3*H*Nh, T)
        qkv = qkv.reshape(B, self.num_head, self.hidden_size*3, T).mT  # (B, H, T, 3*C)
        Q, K, V = torch.split(qkv, self.hidden_size, dim=-1)  # each: (B, H, T, C)

        # pad sequence length to multiple of window size
        Q, pad_len_q = self._pad_to_window(Q, ws)
        K, _ = self._pad_to_window(K, ws)
        V, _ = self._pad_to_window(V, ws)

        T_pad = Q.shape[2]
        num_windows = T_pad // ws

        # reshape to (B*H*num_windows, ws, C)
        def reshape_windows(x):
            return x.reshape(B, self.num_head, num_windows, ws, self.hidden_size) \
                    .permute(0,1,2,3,4).reshape(B*self.num_head*num_windows, ws, self.hidden_size)

        Qw = reshape_windows(Q)
        Kw = reshape_windows(K)
        Vw = reshape_windows(V)

        Aw = F.scaled_dot_product_attention(Qw, Kw, Vw, dropout_p=self.attention_drop, is_causal=False)  # (B*H*Wn, ws, C)

        # restore to (B, H, T_pad, C)
        Aw = Aw.reshape(B, self.num_head, num_windows, ws, self.hidden_size) \
               .reshape(B, self.num_head, T_pad, self.hidden_size)
        if pad_len_q > 0:
            Aw = Aw[:, :, :T, :]

        out = Aw.mT.reshape(B, -1, T)  # (B, H*C, T)
        out = self.proj(out)
        out = out + input

        gate, z = self.MLP(out).chunk(2, dim=1)
        out = out + self.MLP_output(F.silu(gate) * z)

        # return tuple for interface compatibility
        return out, (None, None)
    
class ConvActNorm1d(nn.Module):
    def __init__(self, in_channel, hidden_channel, kernel=7, causal=False):
        super(ConvActNorm1d, self).__init__()
        
        self.in_channel = in_channel
        self.kernel = kernel
        self.causal = causal
        if not causal:
            self.conv = nn.Sequential(nn.Conv1d(in_channel, in_channel, kernel, padding=(kernel-1)//2, groups=in_channel),
                                      RMSNorm(in_channel),
                                      nn.Conv1d(in_channel, hidden_channel, 1),
                                      nn.SiLU(),
                                      nn.Conv1d(hidden_channel, in_channel, 1)
                                     )
        else:
            self.conv = nn.Sequential(nn.Conv1d(in_channel, in_channel, kernel, padding=kernel-1, groups=in_channel),
                                      RMSNorm(in_channel),
                                      nn.Conv1d(in_channel, hidden_channel, 1),
                                      nn.SiLU(),
                                      nn.Conv1d(hidden_channel, in_channel, 1)
                                     )
        
    def forward(self, input):
        
        output = self.conv(input)
        if self.causal:
            output = output[...,:-self.kernel+1]
        return input + output

class ICB(nn.Module):
    def __init__(self, in_channel, kernel=7, causal=False):
        super(ICB, self).__init__()
        
        self.blocks = nn.Sequential(ConvActNorm1d(in_channel, in_channel*4, kernel, causal=causal),
                                    ConvActNorm1d(in_channel, in_channel*4, kernel, causal=causal),
                                    ConvActNorm1d(in_channel, in_channel*4, kernel, causal=causal)
                                    )
        
    def forward(self, input):
        
        return self.blocks(input)

class BSNet(nn.Module):
    def __init__(self, feature_dim, kernel=7):
        super(BSNet, self).__init__()

        self.feature_dim = feature_dim
        # Replace global attention with Window-based MSA to reduce complexity (along band axis)
        # window is on the band dimension (nband). Tune as needed.
        self.band_net = WMSA1D(self.feature_dim, self.feature_dim, num_head=8, window=8)
        self.seq_net = ICB(self.feature_dim, kernel=kernel)

    def forward(self, input):
        # input shape: B, nband, N, T
        B, nband, N, T = input.shape

        # band comm: attention along band axis using W-MSA
        band_input = input.permute(0, 3, 2, 1).reshape(B * T, -1, nband)
        band_output, _ = self.band_net(band_input)
        band_output = band_output.reshape(B, T, -1, nband).permute(0, 3, 2, 1)

        # sequence modeling along time
        output = self.seq_net(band_output.reshape(B * nband, -1, T)).reshape(B, nband, -1, T)

        return output
    
class Apollo(BaseModel):
    def __init__(
        self, 
        sr: int,
        win: int,
        feature_dim: int,
        layer: int
    ):
        super().__init__(sample_rate=sr)
        
        self.sr = sr
        # self.win = int(sr * win // 1000)
        self.win = int(sr * win // 2000)
        self.stride = self.win // 2
        self.enc_dim = self.win // 2 + 1
        self.feature_dim = feature_dim
        self.eps = torch.finfo(torch.float32).eps

        # 80 bands
        # bandwidth = int(self.win / 160)
        # self.band_width = [bandwidth]*79
        bandwidth = int(self.win / 80)  # Instead of 160
        self.band_width = [bandwidth]*39  # Instead of 79
        self.band_width.append(self.enc_dim - np.sum(self.band_width))
        self.nband = len(self.band_width)
        print(self.band_width, self.nband)

        self.BN = nn.ModuleList([])
        for i in range(self.nband):
            self.BN.append(nn.Sequential(RMSNorm(self.band_width[i]*2+1),
                                         nn.Conv1d(self.band_width[i]*2+1, self.feature_dim, 1))
                          )

        self.net = []
        for _ in range(layer):
            self.net.append(BSNet(self.feature_dim))
        self.net = nn.Sequential(*self.net)
        
        self.output = nn.ModuleList([])
        for i in range(self.nband):
            self.output.append(nn.Sequential(RMSNorm(self.feature_dim),
                                                 nn.Conv1d(self.feature_dim, self.band_width[i]*4, 1),
                                                 nn.GLU(dim=1)
                                                )
                                  )

    def spec_band_split(self, input):

        B, nch, nsample = input.shape

        spec = torch.stft(input.view(B*nch, nsample), n_fft=self.win, hop_length=self.stride, 
                          window=torch.hann_window(self.win).to(input.device), return_complex=True)

        subband_spec = []
        subband_spec_norm = []
        subband_power = []
        band_idx = 0
        for i in range(self.nband):
            this_spec = spec[:,band_idx:band_idx+self.band_width[i]]
            subband_spec.append(this_spec)  # B, BW, T
            subband_power.append((this_spec.abs().pow(2).sum(1) + self.eps).sqrt().unsqueeze(1))  # B, 1, T
            subband_spec_norm.append(torch.complex(this_spec.real / subband_power[-1], this_spec.imag / subband_power[-1]))  # B, BW, T
            band_idx += self.band_width[i]
        subband_power = torch.cat(subband_power, 1)  # B, nband, T

        return subband_spec_norm, subband_power

    def feature_extractor(self, input):
        
        subband_spec_norm, subband_power = self.spec_band_split(input)
        
        # normalization and bottleneck
        subband_feature = []
        for i in range(self.nband):
            concat_spec = torch.cat([subband_spec_norm[i].real, subband_spec_norm[i].imag, torch.log(subband_power[:,i].unsqueeze(1))], 1)
            subband_feature.append(self.BN[i](concat_spec))
        subband_feature = torch.stack(subband_feature, 1)  # B, nband, N, T

        return subband_feature
        
    def forward(self, input):

        B, nch, nsample = input.shape

        subband_feature = self.feature_extractor(input)
        feature = self.net(subband_feature)

        est_spec = []
        for i in range(self.nband):
            this_RI = self.output[i](feature[:,i]).view(B*nch, 2, self.band_width[i], -1)
            est_spec.append(torch.complex(this_RI[:,0], this_RI[:,1]))
        est_spec = torch.cat(est_spec, 1)
        output = torch.istft(est_spec, n_fft=self.win, hop_length=self.stride, 
                             window=torch.hann_window(self.win).to(input.device), length=nsample).view(B, nch, -1)
        
        return output
    
    def get_model_args(self):
        model_args = {"n_sample_rate": 2}
        return model_args
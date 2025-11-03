from layers.RevIN import RevIN
from layers.cross_channel_Transformer import Trans_ours
from typing import Callable, Optional
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from layers.PatchTST_layers import *

from layers.channel_mask import channel_mask_generator


class detec_model(nn.Module):
    def __init__(self, configs, **kwargs):
        super(detec_model, self).__init__()

        self.revin_layer = RevIN(configs.c_in, affine=configs.detec_affine, subtract_last=configs.detec_subtract_last)
        self.c_in = configs.c_in
        self.d_model = configs.detec_d_model
        self.cf_dim = configs.detec_cf_dim
        self.d_ff = configs.detec_d_ff
        self.n_heads = configs.detec_n_heads
        self.head_dim = configs.detec_head_dim
        self.dropout = nn.Dropout(configs.detec_dropout)
        self.attn_dropout = configs.detec_attn_dropout
        self.head_dropout = configs.detec_head_dropout
        self.intra_e_layers = configs.detec_intra_e_layers
        self.inter_e_layers = configs.detec_inter_e_layers
        self.regular_lambda = configs.regular_lambda
        self.temperature = configs.detec_temperature
        # Patching
        self.patch_len = configs.detec_patch_len
        self.patch_stride = configs.detec_patch_stride
        self.seq_len = configs.detec_seq_len
        self.horizon = self.seq_len
        patch_num = int((self.seq_len - self.patch_len) / self.patch_stride + 1)
        self.norm = nn.LayerNorm(self.patch_len)
        # print("depth=",cf_depth)
        # Backbone

        self.W_P = nn.Linear(self.patch_len, self.d_model)

        self.W_pos = positional_encoding(pe='zeros', learn_pe=True, q_len=self.seq_len, d_model=self.d_model)

        self.in_vars_tranformer = TSTEncoder(self.seq_len, self.d_model, self.n_heads, d_k=None, d_v=None, d_ff=self.d_ff,
                                             norm='BatchNorm',
                                             attn_dropout=self.attn_dropout, dropout=configs.detec_dropout,
                                             pre_norm=False, activation="gelu", res_attention=False, n_layers=self.intra_e_layers,
                                             store_attn=False)

        self.re_attn = True
        self.mask_generator = channel_mask_generator(patch_len=self.d_model, n_vars=self.c_in)
        self.between_vars_transformer = Trans_ours(dim=self.d_model, depth=self.inter_e_layers, n_heads=self.n_heads,
                                                   mlp_dim=self.d_ff,
                                                   dim_head=self.head_dim, dropout=configs.detec_dropout,
                                                   patch_dim=self.d_model,
                                                   horizon=self.d_model, d_model=self.d_model,
                                                   regular_lambda=self.regular_lambda, temperature=self.temperature)

        # Head
        self.head_nf_f = self.d_model * patch_num
        self.n_vars = self.c_in
        self.individual = configs.individual

        self.ircom = nn.Linear(self.seq_len, self.seq_len)

        # break up R&I:
        self.get_r = nn.Linear(self.d_model, self.d_model)
        self.get_i = nn.Linear(self.d_model, self.d_model)

        self.head_f = Flatten_Head(self.individual, self.n_vars, self.head_nf_f, self.seq_len,
                                    head_dropout=self.head_dropout)


    def forward(self, z):
        z = self.revin_layer(z, 'norm')

        z = z.permute(0, 2, 1)

        # do patching
        z = z.unfold(dimension=-1, size=self.patch_len,
                       step=self.patch_stride)

        # model shape
        batch_size = z.shape[0]
        c_in = z.shape[1]
        patch_num = z.shape[2]
        patch_len = z.shape[3]

        z = self.W_P(z)
        z = self.dropout(z)

        z = torch.reshape(z, (batch_size * c_in, patch_num, z.shape[-1]))

        z = self.in_vars_tranformer(z)

        z = z.view(batch_size, c_in, patch_num, -1).transpose(1, 2).reshape(batch_size * patch_num, c_in, -1)

        channel_mask = self.mask_generator(z)

        z, dcloss = self.between_vars_transformer(z, channel_mask)

        z = torch.reshape(z, (batch_size, patch_num, c_in, z.shape[-1]))

        z = z.permute(0, 2, 1, 3)

        z = self.head_f(z)

        # denorm
        z = z.permute(0, 2, 1)

        z = self.revin_layer(z, 'denorm')

        return z, dcloss


class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, seq_len, head_dropout=0):
        super().__init__()

        self.individual = individual
        self.n_vars = n_vars

        self.flatten = nn.Flatten(start_dim=-2)
        self.linear1 = nn.Linear(nf, nf)
        self.linear2 = nn.Linear(nf, nf)
        self.linear3 = nn.Linear(nf, nf)
        self.linear4 = nn.Linear(nf, seq_len)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]

        x = self.flatten(x)
        x = F.relu(self.linear1(x)) + x
        x = F.relu(self.linear2(x)) + x
        x = F.relu(self.linear3(x)) + x
        x = self.linear4(x)

        return x

class TSTEncoder(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=None,
                        norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                        res_attention=False, n_layers=1, pre_norm=False, store_attn=False):
        super().__init__()

        self.layers = nn.ModuleList([TSTEncoderLayer(q_len, d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                                      attn_dropout=attn_dropout, dropout=dropout,
                                                      activation=activation, res_attention=res_attention,
                                                      pre_norm=pre_norm, store_attn=store_attn) for i in range(n_layers)])
        self.res_attention = res_attention

    def forward(self, src:Tensor):
        output = src
        for mod in self.layers:
            output = mod(output)
        return output

class TSTEncoderLayer(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=256, store_attn=False,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True, activation="gelu", res_attention=False, pre_norm=False):
        super().__init__()
        assert not d_model%n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v


        self.res_attention = res_attention
        self.self_attn = _MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention)

        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn

    def forward(self, src:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None) -> Tensor:

        # Multi-Head attention

        src2, attn = self.self_attn(src, src, src, key_padding_mask=key_padding_mask, attn_mask=attn_mask)

        # Add & Norm
        src = src + self.dropout_attn(src2)
        src = self.norm_attn(src)

        src2 = self.ff(src)
        # Add & Norm
        src = src + self.dropout_ffn(src2)
        src = self.norm_ffn(src)

        return src

class _MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, res_attention=False, attn_dropout=0., proj_dropout=0., qkv_bias=True, lsa=False):

        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        # Scaled Dot-Product Attention (multiple heads)
        self.res_attention = res_attention
        self.sdp_attn = _ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout, res_attention=self.res_attention, lsa=lsa)

        # Project output
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))


    def forward(self, Q:Tensor, K:Optional[Tensor]=None, V:Optional[Tensor]=None, prev:Optional[Tensor]=None,
                key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):

        bs = Q.size(0)
        if K is None: K = Q
        if V is None: V = Q

        # Linear (+ split in multiple heads)
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1,2)
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0,2,3,1)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1,2)
        # Apply Scaled Dot-Product Attention (multiple heads)
        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, prev=prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            output, attn_weights = self.sdp_attn(q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask)

        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v)
        output = self.to_out(output)

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights


class _ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=False, lsa=False):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa

    def forward(self, q:Tensor, k:Tensor, v:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):

        attn_scores = torch.matmul(q, k) * self.scale

        if prev is not None: attn_scores = attn_scores + prev

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask

        if key_padding_mask is not None:
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        output = torch.matmul(attn_weights, v)

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights
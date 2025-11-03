from torch import nn, einsum
from einops import rearrange
import math, torch
from .ch_discover_loss import DynamicalContrastiveLoss


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class c_Attention(nn.Module):
    def __init__(self, dim, n_heads, dim_head, dropout=0.8, regular_lambda=0.3, temperature=0.1):
        super().__init__()
        self.dim_head = dim_head
        self.heads = n_heads
        self.d_k = math.sqrt(self.dim_head)
        inner_dim = dim_head * n_heads
        self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(dim, inner_dim)
        self.to_k = nn.Linear(dim, inner_dim)
        self.to_v = nn.Linear(dim, inner_dim)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
        self.dynamicalContranstiveLoss = DynamicalContrastiveLoss(k=regular_lambda, temperature=temperature)

    def forward(self, x, attn_mask=None):       #x : [bs*patch_num, n_nars, cf_dim]
        h = self.heads
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        scale = 1 / self.d_k

        q = rearrange(q, 'b n (h d) -> b h n d', h=h)       # q : [bs*patch_num, 1, n_nars, inner_dim]
        k = rearrange(k, 'b n (h d) -> b h n d', h=h)       # k : [bs*patch_num, 1, n_nars, inner_dim]
        v = rearrange(v, 'b n (h d) -> b h n d', h=h)       # v : [bs*patch_num, 1, n_nars, inner_dim]

        dynamical_contrastive_loss = None

        scores = einsum('b h i d, b h j d -> b h i j', q, k)

        q_norm = torch.norm(q, dim=-1, keepdim=True)
        k_norm = torch.norm(k, dim=-1, keepdim=True)
        norm_matrix = torch.einsum('bhid,bhjd->bhij', q_norm, k_norm)
        if attn_mask is not None:
            def _mask(scores, attn_mask):
                large_negative = -math.log(1e10)
                attention_mask = torch.where(attn_mask == 0, large_negative, 0)
                scores = scores * attn_mask.unsqueeze(1) + attention_mask.unsqueeze(1)
                return scores

            masked_scores = _mask(scores, attn_mask)    # masked_scores : [bs*patch_num, 1, n_nars, n_nars]

            dynamical_contrastive_loss = self.dynamicalContranstiveLoss(scores, attn_mask, norm_matrix)
        else:
            masked_scores = scores
            dynamical_contrastive_loss = 0

        attn = self.attend(masked_scores * scale)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')        #  out : [bs*patch_num, n_nars, inner_dim(num_heads*inner_dim)]
        out = self.to_out(out)      # out : [bs*patch_num, n_nars, cf_dim]
        return out, attn, dynamical_contrastive_loss


class c_Transformer(nn.Module):  ##Register the blocks into whole network
    def __init__(self, dim, depth, n_heads, dim_head, mlp_dim, dropout=0.8, regular_lambda=0.3, temperature=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim,
                        c_Attention(dim, n_heads=n_heads, dim_head=dim_head, dropout=dropout, regular_lambda=regular_lambda,
                                    temperature=temperature)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x, attn_mask=None):     # x : [bs*patch_num, n_nars, cf_dim]
        total_loss = 0
        for attn, ff in self.layers:
            x_n, attn, dcloss = attn(x, attn_mask=attn_mask)
            total_loss += dcloss
            x = x + x_n.clone()
            x = x + ff(x).clone()
        dcloss = total_loss / len(self.layers)
        return x, attn, dcloss


class Trans_C(nn.Module):
    def __init__(self, *, dim, depth, n_heads, mlp_dim, dim_head, dropout, patch_dim, horizon, d_model,
                 regular_lambda=0.3, temperature=0.1):
        super().__init__()

        self.dim = dim
        self.patch_dim = patch_dim
        self.to_patch_embedding = nn.Sequential(nn.Linear(patch_dim, dim), nn.Dropout(dropout))
        self.dropout = nn.Dropout(dropout)
        self.transformer = c_Transformer(dim, depth, n_heads, dim_head, mlp_dim, dropout, regular_lambda=regular_lambda,
                                         temperature=temperature)

        self.mlp_head = nn.Linear(dim, d_model)  # horizon)

    def forward(self, x, attn_mask=None):   # x : [bs * patch_num, n_vars, patch_len]
        x = self.to_patch_embedding(x)      # [bs*patch_num, n_vars, patch_len] -> [bs*patch_num, n_vars, cf_dim]
        x, attn, dcloss = self.transformer(x, attn_mask)
        x = self.dropout(x)
        x = self.mlp_head(x).squeeze()
        return x, dcloss  # ,attn

class Trans_ours(nn.Module):
    def __init__(self, *, dim, depth, n_heads, mlp_dim, dim_head, dropout, patch_dim, horizon, d_model,
                 regular_lambda=0.3, temperature=0.1):
        super().__init__()

        self.dim = dim
        self.patch_dim = patch_dim
        self.to_patch_embedding = nn.Sequential(nn.Linear(patch_dim, dim), nn.Dropout(dropout))
        self.dropout = nn.Dropout(dropout)
        self.transformer = c_Transformer(dim, depth, n_heads, dim_head, mlp_dim, dropout,
                                         regular_lambda=regular_lambda,
                                         temperature=temperature)

        self.mlp_head = nn.Linear(dim, d_model)

    def forward(self, x, attn_mask=None):
        x, attn, dcloss = self.transformer(x, attn_mask)
        x = self.dropout(x)
        x = self.mlp_head(x).squeeze()
        return x, dcloss  # ,attn

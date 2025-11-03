import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
import numpy as np


class guide_model(nn.Module):
    def __init__(self, configs):
        super(guide_model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.seq_len
        self.d_model = configs.guide_d_model
        self.d_ff = configs.guide_d_ff
        self.output_attention = configs.guide_output_attention
        self.use_norm = configs.guide_use_norm
        self.e_layers = configs.guide_e_layers
        self.dropout = configs.guide_dropout
        self.n_heads = configs.guide_n_heads
        self.factor = configs.guide_factor
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(self.seq_len, self.d_model, configs.embed, configs.freq,
                                                    self.dropout)
        self.class_strategy = configs.guide_class_strategy
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, self.factor, attention_dropout=self.dropout,
                                      output_attention=self.output_attention), self.d_model, self.n_heads),
                    self.d_model,
                    self.d_ff,
                    dropout=self.dropout,
                    activation='gelu'
                ) for l in range(self.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.d_model)
        )
        self.projector = nn.Linear(self.d_model, self.seq_len, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):

        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc_normalized = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc_normalized, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
            x_enc = x_enc_normalized / stdev

        _, _, N = x_enc.shape

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # B N E -> B N S -> B S N 
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N]

        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out, attns


    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        dec_out, attns = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]
import torch
import torch.nn as nn
from layers.RevIN import RevIN
from layers.Leddam_layer import Leddam_layer

class pred_model(nn.Module):
    def __init__(self, configs):
        super(pred_model, self).__init__()
        self.revin=configs.revin
        self.revin_layer=RevIN(num_features=configs.c_in, affine=False, subtract_last=False)
        self.Leddam_layer=Leddam_layer(configs.c_in,configs.seq_len,configs.d_model,
                       configs.dropout,configs.pe_type,kernel_size=25,n_layers=configs.n_layers)
        
        self.Linear_main = nn.Linear(configs.d_model, configs.seq_len)
        self.Linear_res = nn.Linear(configs.d_model, configs.seq_len)
        self.Linear_main.weight = nn.Parameter(
                (1 / configs.d_model) * torch.ones([configs.seq_len, configs.d_model]))
        self.Linear_res.weight = nn.Parameter(
                (1 / configs.d_model) * torch.ones([configs.seq_len, configs.d_model]))
    def forward(self, inp):

        if self.revin:
            inp=self.revin_layer(inp, 'norm')
        res,main=self.Leddam_layer(inp)
        main_out=self.Linear_main(main.permute(0,2,1)).permute(0,2,1)
        res_out=self.Linear_res(res.permute(0,2,1)).permute(0,2,1)
        pred=main_out+res_out
        if self.revin:
            pred=self.revin_layer(pred, 'denorm')
        return pred

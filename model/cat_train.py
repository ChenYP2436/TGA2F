import torch
import torch.nn as nn
from model import Leddam
from model import MIDformer

class Model(nn.Module):
    def __init__(self, configs, **kwargs):
        super(Model, self).__init__()
        self.args = configs
        self.model_dict = {
            'Leddam': Leddam,
            'MIDformer': MIDformer,
        }
        pred, detec = self.args.model.split('_')
        self.cat_train = configs.cat_train
        if not self.cat_train:
            self.args.detec_seq_len = self.args.seq_len
        self.prediction_layer = self.model_dict[pred].pred_model(self.args).float()
        self.detection_layer = self.model_dict[detec].detec_model(self.args).float()

    def Con_cat(self, x_enc, outputs):
        # concat the forcasting outpus and x_enc
        inputs = torch.cat((x_enc, outputs), dim=1)
        result = self.con_cat(inputs.permute(0, 2, 1))

        return result.permute(0, 2, 1)

    def forward(self, x_enc):  # z: [bs x seq_len x n_vars]

        pred = self.prediction_layer(x_enc)  # [bs, seq_len, n_vars]

        if self.cat_train:
            cat = torch.cat((x_enc, pred), dim=1)   # [bs, 2*seq_len, n_vars]
            rec, dcloss = self.detection_layer(cat)
        else:
            cat = pred
            rec, dcloss = self.detection_layer(cat)

        return pred, rec, dcloss

'''
* @author: EmpyreanMoon
*
* @create: 2024-08-26 10:28
*
* @description: The structure of CATCH
'''

from layers.RevIN import RevIN

# Cell
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import TimeBridge
from model import TimeMixer
from model import PatchTST
from model import DeformableTST
from model import iTransformer
from model import Leddam
from model import TimesNet
from model import ModernTCN
from model import DLinear
from model import ModernDetec
from model import iTransformerDetec
from model import TimesNetDetec
from model import AAA
from model import AAA2
from model import AAA2nomask




class Model(nn.Module):
    def __init__(self, configs, **kwargs):
        super(Model, self).__init__()
        self.args = configs
        self.model_dict = {
            'TimeBridge': TimeBridge,
            'TimeMixer': TimeMixer,
            'PatchTST': PatchTST,
            'DeformableTST': DeformableTST,
            'iTransformer': iTransformer,
            'Leddam': Leddam,
            'DLinear': DLinear,
            'TimesNet': TimesNet,
            'ModernTCN': ModernTCN,
            'ModernDetec': ModernDetec,
            'iTransDetec': iTransformerDetec,
            'TsNetDetec': TimesNetDetec,
            'AAA': AAA,
            'AAA2': AAA2,
            'AAA2nomask': AAA2nomask,
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

        # 预测模块
        pred = self.prediction_layer(x_enc)  # [bs, seq_len, n_vars]



        if self.cat_train:
            cat = torch.cat((x_enc, pred), dim=1)   # [bs, 2*seq_len, n_vars]
            # 检测模块
            rec, dcloss = self.detection_layer(cat)
        else:
            cat = pred
            rec, dcloss = self.detection_layer(cat)


        return pred, rec, dcloss

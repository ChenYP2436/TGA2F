'''
* @author: EmpyreanMoon
*
* @create: 2024-09-02 17:29
*
* @description: the implementation of the dynamical contrastive loss
'''

import torch
# ClusteringLoss：鼓励注意力集中于通道掩码定义的“相关通道对
# RegularLoss：限制掩码矩阵不要退化为全1矩阵（Channel-Dependent 极端情况）

class DynamicalContrastiveLoss(torch.nn.Module):
    def __init__(self, temperature=0.5, k=0.3):     # temperature : 对比学习中的温度系数 / k : RegularLoss 的权重系数（λ₃）
        super(DynamicalContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.k = k

    def _stable_scores(self, scores):
        max_scores = torch.max(scores, dim=-1)[0].unsqueeze(-1)
        stable_scores = scores - max_scores
        return stable_scores

    def forward(self, scores, attn_mask, norm_matrix):
        b = scores.shape[0]
        n_vars = scores.shape[-1]

        cosine = (scores / norm_matrix).mean(1)     # 将每个头的注意力分数归一化为“余弦相似度”，并在所有注意力头上取平均
        pos_scores = torch.exp(cosine / self.temperature) * attn_mask       # “正样本”分数：由掩码决定，只保留掩码为 1 的通道对（即认为相关）

        all_scores = torch.exp(cosine / self.temperature)   # 全部样本分数：包括所有通道对的注意力打分（不论相关与否）

        clustering_loss = -torch.log(pos_scores.sum(dim=-1) / all_scores.sum(dim=-1))       # 损失思想：鼓励“正对”的注意力得分尽可能大，占据更高的比重

        eye = torch.eye(attn_mask.shape[-1]).unsqueeze(0).repeat(b, 1, 1).to(attn_mask.device)      # 构造一个 I 单位矩阵：对角线为 1，其他为 0
        # L1范数表示当前掩码矩阵与单位矩阵（CI策略）的差异, 作用是抑制掩码矩阵退化为全 1（CD策略），保持稀疏、可解释、通道选择性
        regular_loss = 1 / (n_vars * (n_vars - 1)) * torch.norm(eye.reshape(b, -1) - attn_mask.reshape((b, -1)),p=1, dim=-1)
        loss = clustering_loss.mean(1) + self.k * regular_loss

        mean_loss = loss.mean()
        return mean_loss

import torch
import torch.nn as nn


class EviLoss_m66(nn.Module):
    def __init__(self, ):
        super(EviLoss_m66, self).__init__()
        self.ce = nn.BCELoss()
        self.eps = 1e-8

    def forward(self, results, targets):
        alpha, u = results['alpha'], results['u']
        num = targets.size(0)

        S = torch.sum(alpha, dim=1, keepdim=True)
        pred = alpha / S

        totalp = 0
        log_cali = 0

        c = torch.clamp(1 - u, min=self.eps, max=1)
        totaldice = 0

        weight = 1
        for d in range(pred.shape[1]):
            m1 = pred[:, d].view(num, -1)
            m2 = targets[:, d].view(num, -1).float()
            ci = c[:, d].view(num, -1)
            p = torch.clamp(ci * m1 + (1 - ci) * m2, min=self.eps)
            log_p = -(torch.log(p + self.eps) * m2)
            totalp += weight * log_p.sum(-1)

            intersection = (m1 * m2)
            score = 2. * (intersection.sum(1) + 1e-8) / (m1.sum(1) + m2.sum(1) + 1e-8)
            score = (1 - score).sum() / num
            totaldice += score


            log_cali += -weight * torch.log(c[:, d] + self.eps).view(num, -1).sum(-1)


        return totalp.mean(), log_cali.mean(), totaldice

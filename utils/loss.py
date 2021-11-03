"""Custom losses."""
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np
import paddle
__all__ = ['ICNetLoss']

# TODO: optim function
class ICNetLoss(nn.CrossEntropyLoss):
    """Cross Entropy Loss for ICNet"""
    
    def __init__(self, aux_weight=0.4, ignore_index=-1):
        super(ICNetLoss, self).__init__(ignore_index=ignore_index, axis=1)
        self.aux_weight = aux_weight
    def forward(self, preds, target):
        # preds, target = tuple(inputs)
        inputs1 = tuple(list(preds) + [target])  
        pred, pred_sub4, pred_sub8, pred_sub16, target = tuple(inputs1)
        # [batch, H, W] -> [batch, 1, H, W]
        target1 = target.unsqueeze(1).astype('float64')
        target1_sub4 = F.interpolate(x = target1, size = pred_sub4.shape[2:], mode='bilinear', align_corners=True).squeeze(1).astype('int64')
        target1_sub8 = F.interpolate(target1, pred_sub8.shape[2:], mode='bilinear', align_corners=True).squeeze(1).astype('int64')
        target1_sub16 = F.interpolate(target1, pred_sub16.shape[2:], mode='bilinear', align_corners=True).squeeze(
            1).astype('int64')
        loss1 = super(ICNetLoss, self).forward(pred_sub4, target1_sub4)
        loss2 = super(ICNetLoss, self).forward(pred_sub8, target1_sub8)
        loss3 = super(ICNetLoss, self).forward(pred_sub16, target1_sub16)
        return loss1 + loss2 * self.aux_weight + loss3 * self.aux_weight


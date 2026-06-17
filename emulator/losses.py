import torch
import torch.nn as nn


class MSE_PixelFlow(nn.Module):
    def forward(self, model_output, target, seqlen_list_q, **kwargs):
        loss = (model_output.float() - target.float()) ** 2
        loss_split = torch.split(loss, seqlen_list_q, dim=0)
        return torch.stack([x.mean() for x in loss_split])

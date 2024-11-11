import torch
from torch import nn

from utils.register import Register

@Register(name='awl')
class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss
    参考:
    1. https://zhuanlan.zhihu.com/p/367881339
    2. https://arxiv.org/pdf/1805.06334
    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        assert len(x) == len(self.params)
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum
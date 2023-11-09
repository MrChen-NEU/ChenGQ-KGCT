import torch.nn as nn


class MateModel(nn.Module):
    def __init__(self):
        super(MateModel, self).__init__()

    def train_epoch(self,
                    batch_size=128):
        # 在这个基类中，方法的实现被抛出了一个 NotImplementedError 异常，这意味着子类必须提供具体的训练逻辑。
        raise NotImplementedError

    def test(self,
             batch_size=128,
             dataset='valid',
             metric_list=None,
             filter_out=False):
        raise NotImplementedError

    def get_config(self):
        raise NotImplementedError

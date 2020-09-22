import torch.nn as nn

import torch.nn.functional as F


class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    we use the correct gelu
    """

    def forward(self, x):
        return F.gelu(x)

import math

import torch
from torch.nn import init

from .base import BaseModel
from .bert_modules.bert import BERTTrainModel
from .bert_modules.bert_content import BERTContent

import torch.nn as nn

from .bert_modules.utils import GELU


class LinearTransformLayer(nn.Module):

    def __init__(self, bert: BERTTrainModel, voc_size_items: int):
        super().__init__()

        self.linear_transform = nn.Linear(bert.hidden, voc_size_items)

    def forward(self, input):
        return self.linear_transform(input)


class BERT4RecTransformLayer(nn.Module):

    def __init__(self, bert: BERTTrainModel, voc_size_items: int):
        super().__init__()
        self.bert = bert
        self.transform = nn.Linear(self.bert.hidden, self.bert.hidden)
        self.out_bias = nn.Parameter(torch.Tensor(voc_size_items))
        self.gelu = GELU()

        self._init_bias()

    def forward(self, input):
        transformed = self.gelu(self.transform(input))
        transposed_embedding = torch.transpose(self.bert.get_item_embedding().weight, 0, 1)
        output = torch.matmul(transformed, transposed_embedding)
        output += self.out_bias
        return output

    def _init_bias(self):
        # Initialization as done for the bias in nn.Layer
        fan_in = self.bert.get_item_embedding().weight.size(0)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.out_bias, -bound, bound)


TRANSFORM_LAYERS = {
    'linear': LinearTransformLayer,
    'bert4rec': BERT4RecTransformLayer
}


def build_transform_layer(transform_layer_type: str, bert: BERTTrainModel, voc_size_items: int):
    transform_model = TRANSFORM_LAYERS[transform_layer_type]
    return transform_model(bert, voc_size_items)


class BERTContentModel(BaseModel):

    def __init__(self, args):
        super().__init__(args)
        self.bert = BERTContent(args)
        voc_size_items = args.num_items + 2 # mask and padding
        self.transform = build_transform_layer(args.transform_layer, self.bert, voc_size_items)

    @classmethod
    def code(cls):
        return 'bert_content'

    def forward(self, x, content):
        x = self.bert(x, content)
        output = self.transform(x)
        return output

from .base import BaseModel
from .bert_content import build_transform_layer
from .bert_modules.bert import BERT


class BERTModel(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.bert = BERT(args)
        voc_size_items = args.num_items + 2 # mask, padding
        self.transform = build_transform_layer(args.transform_layer, self.bert, voc_size_items)

    @classmethod
    def code(cls):
        return 'bert'

    def forward(self, x):
        x = self.bert(x)
        output = self.transform(x)

        return output

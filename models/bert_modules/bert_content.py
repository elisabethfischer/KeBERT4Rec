from torch import nn as nn

from models.bert_modules.bert import BERTTrainModel
from models.bert_modules.embedding.content import get_content_encoding_builder


class BERTContent(BERTTrainModel):
    def __init__(self, args):
        super().__init__(args, embedding_dropout=0.0)
        content_encoding_type = args.content_encoding_type
        content_embedding_build_func = get_content_encoding_builder(content_encoding_type)

        num_content = args.num_content[0]
        content_size = num_content + 2

        self.content_encoding = content_embedding_build_func(vocab_size=content_size, embed_size=self.hidden, args=args)
        dropout = args.bert_dropout
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, content):
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x)

        content_embedding = self.content_encoding(content)

        x = self.dropout(x + content_embedding)

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        return x

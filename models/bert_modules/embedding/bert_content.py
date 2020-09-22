import torch.nn as nn
from .token import TokenEmbedding
from .position import PositionalEmbedding


class BERTContentEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        3. ContentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)

        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, vocab_size, content_size, embed_size, max_len, dropout=0.1):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        self.position = PositionalEmbedding(max_len=max_len, d_model=embed_size)
        self.content = TokenEmbedding(vocab_size=content_size, embed_size=embed_size)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

    def forward(self, sequence, c_sequence):
        # content emb is B x Max_size x max_num_categories_in_batch x emb_dim and the sum converts it to B x MS x emb_dim
        content_embedding = self.content(c_sequence)
        content_embedding = content_embedding.sum(dim=2)
        x = (self.token(sequence) + self.position(sequence)) + content_embedding
        return self.dropout(x)

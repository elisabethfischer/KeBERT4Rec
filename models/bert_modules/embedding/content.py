import torch
from torch import nn
from transformers import BertConfig

from .token import TokenEmbedding


def _build_content_embedding(vocab_size, embed_size, args):
    return ContentEmbedding(vocab_size, embed_size)


def _build_simple_encoding(vocab_size, embed_size, args):
    return SimpleUpscaler(vocab_size, embed_size)


def get_content_encoding_builder(content_embedding_type):
    return {
        'concat_embedding': _build_content_embedding,
        'simple_embedding': _build_simple_encoding,
    }[content_embedding_type]


class ContentEmbedding(nn.Module):

    def __init__(self, vocab_size, embed_size):
        super().__init__()
        self.embedding = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)

    def forward(self, content_input):
        content_input = content_input.squeeze()
        return self.embedding(content_input)


class SimpleUpscaler(nn.Module):

    def __init__(self, vocab_size: int, embed_size: int):
        super().__init__()
        self.upscaler = nn.Linear(vocab_size, embed_size)
        self.vocab_size = vocab_size

    def forward(self, content_input):
        # the input is a sequence of content ids without any order
        # so we convert them into a multi-hot encoding
        multi_hot = torch.nn.functional.one_hot(content_input, self.vocab_size).sum(2).float()
        # 0 is the padding category, so zero it out
        multi_hot[:, :, 0] = 0
        return self.upscaler(multi_hot)
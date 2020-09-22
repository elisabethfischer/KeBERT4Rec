from .content_tokenizers import FullStringToOneIdContentTokenizer, MultiHotContentTokenizer

CONTENTTOKENIZERS = \
    {
        'concat_embedding': FullStringToOneIdContentTokenizer,
        'simple_embedding': MultiHotContentTokenizer
    }


def tokenizer_factory(args):
    encoding_type = args.content_encoding_type
    if encoding_type in CONTENTTOKENIZERS:
        return CONTENTTOKENIZERS[encoding_type]
    return None

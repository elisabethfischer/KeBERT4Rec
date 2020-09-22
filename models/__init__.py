from .bert import BERTModel
from .bert_content import BERTContentModel

MODELS = {
    BERTModel.code(): BERTModel,
    BERTContentModel.code(): BERTContentModel
}


def model_factory(args):
    model = MODELS[args.model_code]
    return model(args)

from abc import *
import random


class AbstractDataloader(metaclass=ABCMeta):
    def __init__(self, args, dataset):
        self.args = args
        seed = args.dataloader_random_seed
        self.rng = random.Random(seed)
        self.save_folder = dataset._get_preprocessed_folder_path()
        dataset = dataset.load_dataset()
        self.train = dataset['train']
        self.val = dataset['val']
        self.test = dataset['test']
        self.umap = dataset['umap']
        self.smap = dataset['smap']
        self.tokenizers = dataset['tokenizers']
        self.user_count = len(self.umap)
        self.item_count = len(self.smap)
        content_count = 0
        if len(self.tokenizers) > 0:
            content_count = self.tokenizers[0].num_classes
        self.content_count = content_count

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    @abstractmethod
    def get_pytorch_dataloaders(self):
        pass

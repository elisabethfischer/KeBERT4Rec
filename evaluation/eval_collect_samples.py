from options import args
from models import model_factory
from dataloaders import dataloader_factory
from trainers import trainer_factory
from utils import *
import json


def test():
    export_root = args.test_model_path
    config_file = os.path.join(export_root, "config.json")
    with open(config_file) as json_file:
        data = json.load(json_file)
        for d in data:
            vars(args)[d] = data[d]
    batchsize = 1
    args.batchsize = batchsize
    args.train_batch_size = batchsize
    args.val_batch_size = batchsize
    args.test_batch_size = batchsize
    args.retrieve_test_samples = True
    train_loader, val_loader, test_loader = dataloader_factory(args)
    model = model_factory(args)
    trainer = trainer_factory(args, model, train_loader, val_loader, test_loader, export_root)
    trainer.test()


if __name__ == '__main__':
    test()

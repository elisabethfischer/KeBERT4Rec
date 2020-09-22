from .bert import BERTTrainer
from .bert_content import BERTContentTrainer


TRAINERS = {
    BERTTrainer.code(): BERTTrainer,
    BERTContentTrainer.code(): BERTContentTrainer
}


def trainer_factory(args, model, train_loader, val_loader, test_loader, export_root):
    trainer = TRAINERS[args.trainer_code]
    return trainer(args, model, train_loader, val_loader, test_loader, export_root)

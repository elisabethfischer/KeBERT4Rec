import torch
import torch.utils.data as data_utils

from .base import AbstractDataloader
from .negative_samplers import negative_sampler_factory


class BertContentDataloader(AbstractDataloader):
    def __init__(self, args, dataset):
        super().__init__(args, dataset)
        args.num_items = len(self.smap)
        args.num_content = [x.num_classes for x in self.tokenizers]
        self.max_len = args.bert_max_len
        self.mask_prob = args.bert_mask_prob
        self.CLOZE_MASK_TOKEN = self.item_count + 1
        self.CLOZE_MASK_CONTENT = self.content_count + 1
        self.max_content_seq_len = [t.max_seq_len for t in self.tokenizers]
        code = args.train_negative_sampler_code
        train_negative_sampler = negative_sampler_factory(code, self.train, self.val, self.test,
                                                          self.user_count, self.item_count,
                                                          args.train_negative_sample_size,
                                                          args.train_negative_sampling_seed,
                                                          self.save_folder)
        code = args.test_negative_sampler_code
        test_negative_sampler = negative_sampler_factory(code, self.train, self.val, self.test,
                                                         self.user_count, self.item_count,
                                                         args.test_negative_sample_size,
                                                         args.test_negative_sampling_seed,
                                                         self.save_folder)

        self.train_negative_samples = train_negative_sampler.get_negative_samples()
        self.test_negative_samples = test_negative_sampler.get_negative_samples()

    @classmethod
    def code(cls):
        return 'bert_content'

    def get_pytorch_dataloaders(self):
        train_loader = self._get_train_loader()
        val_loader = self._get_val_loader()
        test_loader = self._get_test_loader()
        return train_loader, val_loader, test_loader

    def _get_train_loader(self):
        dataset = self._get_train_dataset()
        dataloader = data_utils.DataLoader(dataset, batch_size=self.args.train_batch_size,
                                           shuffle=True, pin_memory=True)
        return dataloader

    def _get_train_dataset(self):
        dataset = BertTrainContentDataset(self.train, self.max_len, self.mask_prob, self.CLOZE_MASK_TOKEN,
                                          self.CLOZE_MASK_CONTENT, self.item_count, self.content_count, self.rng,
                                          content_max_len=self.max_content_seq_len[0])
        return dataset

    def _get_val_loader(self):
        return self._get_eval_loader(mode='val')

    def _get_test_loader(self):
        return self._get_eval_loader(mode='test')

    def _get_eval_loader(self, mode):
        batch_size = self.args.val_batch_size if mode == 'val' else self.args.test_batch_size
        dataset = self._get_eval_dataset(mode)
        dataloader = data_utils.DataLoader(dataset, batch_size=batch_size,
                                           shuffle=False, pin_memory=True)
        return dataloader

    def _get_eval_dataset(self, mode):
        answers = self.val if mode == 'val' else self.test
        dataset = BertEvalContentDataset(self.train, answers, self.max_len, self.CLOZE_MASK_TOKEN,
                                         self.CLOZE_MASK_CONTENT, self.test_negative_samples,
                                         content_max_len=self.max_content_seq_len[0])
        return dataset


class BertTrainContentDataset(data_utils.Dataset):
    def __init__(self, u2seq, max_len, mask_prob, mask_token, content_mask_token, num_items, num_contents, rng,
                 content_max_len=1):
        self.u2seq = u2seq
        self.users = sorted(self.u2seq.keys())
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.mask_token = mask_token
        self.content_mask_token = content_mask_token
        self.num_items = num_items
        self.num_contents = num_contents
        self.rng = rng
        self.content_max_len = content_max_len

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self._getseq(user)

        tokens = []
        labels = []
        content = []
        for s in range(len(seq[0])):
            prob = self.rng.random()
            if prob < self.mask_prob:
                prob /= self.mask_prob

                if prob < 0.8:
                    tokens.append(self.mask_token)
                    content.append([self.content_mask_token])
                elif prob < 0.9:
                    tokens.append(self.rng.randint(1, self.num_items))
                    content.append([self.rng.randint(1, self.num_contents)])
                else:
                    tokens.append(seq[0][s])
                    content.append(seq[1][s])

                labels.append(seq[0][s])
            else:
                tokens.append(seq[0][s])
                labels.append(0)
                content.append(seq[1][s])

        tokens = tokens[-self.max_len:]
        content = content[-self.max_len:]
        labels = labels[-self.max_len:]

        mask_len = self.max_len - len(tokens)

        tokens = [0] * mask_len + tokens
        labels = [0] * mask_len + labels
        content = [[0]] * mask_len + content

        content = [s + [0] * (self.content_max_len - len(s)) for s in content]
        content = torch.as_tensor(content)

        return torch.LongTensor(tokens), content, torch.LongTensor(labels)

    def _getseq(self, user):
        return self.u2seq[user]


class BertEvalContentDataset(data_utils.Dataset):
    def __init__(self, u2seq, u2answer, max_len, mask_token, content_mask_token, negative_samples, content_max_len=1):
        self.u2seq = u2seq
        self.users = sorted(self.u2seq.keys())
        self.u2answer = u2answer
        self.max_len = max_len
        self.mask_token = mask_token
        self.content_mask_token = content_mask_token
        self.negative_samples = negative_samples
        self.content_max_len = content_max_len

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self.u2seq[user]
        answer = self.u2answer[user]
        negs = self.negative_samples[user]

        candidates = answer[0] + negs
        labels = [1] * len(answer[0]) + [0] * len(negs)

        inputs = []

        nseq = seq[0] + [self.mask_token]
        nseq = nseq[-self.max_len:]
        padding_len = self.max_len - len(nseq)
        nseq = [0] * padding_len + nseq
        inputs.append(torch.LongTensor(nseq))

        nseq = seq[1] + [[self.content_mask_token]]
        nseq = nseq[-self.max_len:]
        padding_len = self.max_len - len(nseq)
        nseq = [[0]] * padding_len + nseq

        content = [s + [0] * (self.content_max_len - len(s)) for s in nseq]
        content = torch.as_tensor(content)
        inputs.append(content)

        return inputs[0], inputs[1], torch.LongTensor(candidates), torch.LongTensor(labels)

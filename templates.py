def set_template(args):
    if args.template is None:
        return
    
    else:
        content_bert = args.template.startswith("train_content_bert")
        debug = args.template == "debug"
        bert = args.template.startswith('train_bert')

        if content_bert or bert or debug:
            args.mode = "train"

            dataset_to_use = args.dataset_code
            args.dataset_code = dataset_to_use if args.dataset_code else input('ml-1m, ml-20m: ')
            args.min_rating = 0 if args.dataset_code == 'ml-1m' else 4
            args.min_uc = 5
            args.min_sc = 0
            args.split = 'leave_one_out'
            transform_layer = args.transform_layer
            args.transform_layer = transform_layer if transform_layer else 'bert4rec'

            batch_size = args.train_batch_size
            batch = batch_size if batch_size else 128
            args.train_batch_size = batch
            args.val_batch_size = batch
            args.test_batch_size = batch

            args.train_negative_sampler_code = 'random'
            args.train_negative_sample_size = 0
            args.train_negative_sampling_seed = 0
            args.test_negative_sampler_code = 'random'
            args.test_negative_sample_size = 100
            args.test_negative_sampling_seed = 98765

            device = args.device
            args.device = device if device else 'cuda' if not debug else "cpu"
            args.num_gpu = 1
            args.device_idx = '0'
            args.optimizer = 'Adam'
            args.lr = 0.0001
            args.enable_lr_schedule = True
            args.decay_step = 25
            args.gamma = 1.0
            epochs = args.num_epochs
            args.num_epochs = epochs if epochs else 50 if args.dataset_code in ['ml-1m'] else 200
            args.metric_ks = [1, 5, 10, 20, 50, 100]
            args.best_metric = 'NDCG@10'

            args.model_init_seed = 0

            args.bert_dropout = 0.2
            args.bert_hidden_units = 64
            args.bert_mask_prob = 0.2
            args.bert_max_len = 200
            args.bert_num_blocks = 2
            args.bert_num_heads = 2

        if debug:
            args.num_epochs = 2
        if content_bert:
            args.dataloader_code = 'bert_content'
            args.trainer_code = 'bert_content'
            args.model_code = 'bert_content'

            content_encoding_type = args.content_encoding_type
            args.content_encoding_type = content_encoding_type if content_encoding_type else 'concat_embedding'
        elif bert:
            args.dataloader_code = 'bert'
            args.trainer_code = 'bert'
            args.model_code = 'bert'



# Introduction

This repository implements models from the following two papers:

> **BERT4Rec: Sequential Recommendation with BERT (Sun et al.)** 

> **KeBERT4Rec: Integrating keywords into BERT4Rec for Sequential Recommendation**  

and lets you train them on MovieLens-1m and MovieLens-20m.

The implementation of BERT4Rec is based on:
> **https://github.com/jaywonchung/BERT4Rec-VAE-Pytorch (66f0853) from Jaewon Chung**

# Usage

## Overall

Run `main.py` with arguments to train and/or test you model. There is a template file 'templates.py' available.

```bash
python main.py --template train_bert
```

## Examples

1. Train BERT4Rec on ML-20m and run test set inference after training

   ```bash
   python main.py --dataset_code=ml-20m --template=train_bert --experiment_description=ml-20m_baseline --train_batch_size=64"
   ```

2. Train BERT4Rec on ML-20m with merge embedding

      ```bash
      python main.py --dataset_code=ml-20m --template=train_content_bert 
   --experiment_description=ml-20m_merge_embedding 
   --train_batch_size=64      ```
   
3. Train BERT4Rec on ML-20m with multi-hot embedding

      ```bash
      python main.py --dataset_code=ml-20m --template=train_content_bert 
   --content_encoding_type=simple_embedding --experiment_description=ml-20m_simple_embedding 
   --train_batch_size=64      ```
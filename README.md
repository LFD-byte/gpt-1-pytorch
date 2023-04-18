# OpenAI GPT

PyTorch Implementation of OpenAI GPT

<p align="center"><img width= 70 src="https://pytorch.org/assets/images/logo-icon.svg"></p>

## Quick Start
### 0. Install dependencies
PreNLP is Preprocessing Library for Natural Language Processing. It provides sentencepiece tokenizer.
```
$ pip3 install prenlp
```
<br>

### 1. Setup input pipeline

#### Building vocab based on your corpus
```
$ python vocab.py --corpus <YOUR_CORPUS> --prefix <VOCAB_NAME> --vocab_size <YOUR_VOCAB_SIZE>
```

you can train WikiText-103 corpus using below command, and build vocab based on this.
```
$ python vocab.py --corpus data/wikitext-103/wiki.test --prefix wiki.test
```
<br>

### 2. Unsupervised pre-training
```
$ python main.py --train_corpus <TRAIN_CORPUS> --vocab_file <VOCAB_FILE> --pretrained_sp_model <PRETRAINED_SP_MODEL> --pretrain
```
```
$ python main.py \
    --train_corpus data/wikitext-103/wiki.test \
    --vocab_file model_hub/wiki.test.vocab \
    --pretrained_sp_model model_hub/wiki.test.model \
    --epochs 100 \
    --batch_size 64 \
    --n_attn_heads 4 \
    --n_layers 4 \
    --output_model_prefix wiki.test \
    --pretrain
```
<br>

### 3. Supervised fine-tuning
```
$ python main.py --train_corpus <TRAIN_CORPUS> --test_corpus <TEST_CORPUS>  --vocab_file <VOCAB_FILE> --pretrained_sp_model <PRETRAINED_SP_MODEL> --pretrained_model <PRETRAINED_MODEL> --finetune --do_eval
```
```
$ python main.py \
  --train_corpus data/aclImdb/imdb.train \
  --test_corpus data/aclImdb/imdb.test \
  --vocab_file model_hub/wiki.test.vocab \
  --pretrained_sp_model model_hub/wiki.test.model \
  --pretrained_model checkpoints/wiki.test.ep8  \
  --finetune \
  --do_eval
```
<br>

## List of options
You may need to change below argument parameters.
```
$ python main.py -h
usage: main.py [-h] --train_corpus TRAIN_CORPUS --vocab_file VOCAB_FILE
               --pretrained_sp_model PRETRAINED_SP_MODEL [--pretrain]
               [--finetune] [--do_eval] [--test_corpus TEST_CORPUS]
               [--pretrained_model PRETRAINED_MODEL]
               [--output_model_prefix OUTPUT_MODEL_PREFIX]
               [--batch_size BATCH_SIZE] [--max_seq_len MAX_SEQ_LEN]
               [--n_workers N_WORKERS] [--epochs EPOCHS] [--lr LR]
               [--auxiliary_ratio AUXILIARY_RATIO] [--local_rank LOCAL_RANK]
               [--no_cuda] [--distributed] [--hidden HIDDEN]
               [--n_layers N_LAYERS] [--n_attn_heads N_ATTN_HEADS]
               [--embd_dropout EMBD_DROPOUT] [--resid_dropout RESID_DROPOUT]
               [--attn_dropout ATTN_DROPOUT] [--ffn_hidden FFN_HIDDEN]
               [--cached_label_dict CACHED_LABEL_DICT]

optional arguments:
  -h, --help            show this help message and exit
  --train_corpus TRAIN_CORPUS
                        corpus for either pre-train or fine-tune
  --vocab_file VOCAB_FILE
                        pretrained vocabulary
  --pretrained_sp_model PRETRAINED_SP_MODEL
                        pretrained sentencepiece model
  --pretrain
  --finetune
  --do_eval
  --test_corpus TEST_CORPUS
                        corpus for either pre-train or fine-tune evaluation
  --pretrained_model PRETRAINED_MODEL
                        pretrained GPT model path
  --output_model_prefix OUTPUT_MODEL_PREFIX
                        output model name prefix
  --batch_size BATCH_SIZE
                        batch size
  --max_seq_len MAX_SEQ_LEN
                        the maximum size of the input sequence
  --n_workers N_WORKERS
                        the number of workers
  --epochs EPOCHS       the number of epochs
  --lr LR               initial learning rate
  --auxiliary_ratio AUXILIARY_RATIO
                        weight of auxiliary objective
  --local_rank LOCAL_RANK
                        node rank for distributed training
  --no_cuda
  --distributed
  --hidden HIDDEN       the number of expected features in the transformer
                        decoder
  --n_layers N_LAYERS   the number of decoder layers
  --n_attn_heads N_ATTN_HEADS
                        the number of multi-head attention heads
  --embd_dropout EMBD_DROPOUT
                        embedding dropout value
  --resid_dropout RESID_DROPOUT
                        residual dropout value
  --attn_dropout ATTN_DROPOUT
                        attention dropout value
  --ffn_hidden FFN_HIDDEN
                        dimension of the feedforward network
  --cached_label_dict CACHED_LABEL_DICT
```

### References
- [Improving Language Understandingby Generative Pre-Training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)
- [openai / finetune-transformer-lm](https://github.com/openai/finetune-transformer-lm)
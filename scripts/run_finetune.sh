# fine-tuning
python main.py \
  --train_corpus data/aclImdb/imdb.train \
  --test_corpus data/aclImdb/imdb.test \
  --vocab_file model_hub/wiki.test.vocab \
  --pretrained_sp_model model_hub/wiki.test.model \
  --pretrained_model checkpoints/wiki.test.ep8  \
  --finetune \
  --do_eval
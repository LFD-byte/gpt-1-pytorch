# pre-training
python main.py \
    --train_corpus data/wikitext-103/wiki.test \
    --vocab_file model_hub/wiki.test.vocab \
    --pretrained_sp_model model_hub/wiki.test.model \
    --epochs 100 \
    --batch_size 64 \
    --n_attn_heads 4 \
    --n_layers 4 \
    --output_model_prefix wiki.test \
    --pretrain
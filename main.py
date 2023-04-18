import argparse

from torch.utils.data import DataLoader, RandomSampler

from trainer import Trainer
from utils.data_utils import create_examples
from utils.tokenization import PretrainedTokenizer


def main(args):
    print(args)

    # Load pretrained tokenizer
    tokenizer = PretrainedTokenizer(pretrained_model=args.pretrained_sp_model, vocab_file=args.vocab_file)

    # Build DataLoader
    train_dataset = create_examples(args, tokenizer, mode='train')
    train_sampler = RandomSampler(train_dataset)
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size,
                              num_workers=args.n_workers)
    if args.do_eval:
        test_dataset = create_examples(args, tokenizer, mode='test')
        test_sampler = RandomSampler(test_dataset)
        test_loader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size,
                                 num_workers=args.n_workers)

    # Build Trainer
    trainer = Trainer(args=args,
                      train_loader=train_loader,
                      test_loader=test_loader if args.do_eval else None,
                      tokenizer=tokenizer)

    # Train
    for epoch in range(1, args.epochs + 1):
        trainer.train(epoch)
        trainer.save(epoch, args.output_model_prefix)
        if args.do_eval:
            trainer.evaluate(epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_corpus', required=True, type=str, help='corpus for either pre-train or fine-tune')
    parser.add_argument('--vocab_file', required=True, type=str, help='pretrained vocabulary')
    parser.add_argument('--pretrained_sp_model', required=True, type=str, help='pretrained sentencepiece model')
    parser.add_argument('--pretrain', action='store_true')
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--do_eval', action='store_true')

    parser.add_argument('--test_corpus', default=None, type=str,
                        help='corpus for either pre-train or fine-tune evaluation')
    parser.add_argument('--pretrained_model', default=None, type=str, help='pretrained GPT model path')
    parser.add_argument('--output_model_prefix', default='model', type=str, help='output model name prefix')
    # Input parameters
    parser.add_argument('--batch_size', default=8, type=int, help='batch size')
    parser.add_argument('--max_seq_len', default=512, type=int, help='the maximum size of the input sequence')
    parser.add_argument('--n_workers', default=1, type=int, help='the number of workers')
    # Train parameters
    parser.add_argument('--epochs', default=100, type=int, help='the number of epochs')
    parser.add_argument('--lr', default=1.5e-4, type=float, help='initial learning rate')
    parser.add_argument('--auxiliary_ratio', default=.25, type=float, help='weight of auxiliary objective')
    parser.add_argument('--no_cuda', action='store_true')
    # Model parameters
    parser.add_argument('--hidden', default=768, type=int,
                        help='the number of expected features in the transformer decoder')
    parser.add_argument('--n_layers', default=12, type=int, help='the number of decoder layers')
    parser.add_argument('--n_attn_heads', default=12, type=int, help='the number of multi-head attention heads')
    parser.add_argument('--embd_dropout', default=0.1, type=float, help='embedding dropout value')
    parser.add_argument('--resid_dropout', default=0.1, type=float, help='residual dropout value')
    parser.add_argument('--attn_dropout', default=0.1, type=float, help='attention dropout value')
    parser.add_argument('--ffn_hidden', default=3072, type=int, help='dimension of the feedforward network')
    # Others
    parser.add_argument('--cached_label_dict', default='cached_label_dict.json', type=str)

    args = parser.parse_args()

    main(args)

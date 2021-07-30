import os

from transformers import (T5Config, T5ForConditionalGeneration, Trainer,
                          TrainingArguments, HfArgumentParser)

from data import read_parallel_split


def main(training_args, args):
    if not os.path.isdir(args.model_dir):
        os.makedirs(args.model_dir)
        config = T5Config(vocab_size=32100)
        config.decoder_start_token_id = config.pad_token_id
        model = T5ForConditionalGeneration(config)
    else:
        model = T5ForConditionalGeneration.from_pretrained(args.model_dir)

    train_dataset, val_dataset = read_parallel_split(args.parallel_path,
        inverse=args.inverse)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )
    trainer.train()
    trainer.save_model(args.model_dir)


if __name__ == '__main__':
    parser = HfArgumentParser(TrainingArguments)
    parser.add_argument('-m', '--model_dir', required=True,
                        help='Model save/load directory')
    parser.add_argument('-p', '--parallel_path', required=True,
                        help='Path to tokenized parallel corpus')
    parser.add_argument('-i', '--inverse', action='store_true',
                        help='Train for backtranslation (target -> source)')
    training_args, args = parser.parse_args_into_dataclasses()
    main(training_args, args)

import argparse
import glob
import os

from fugashi import Tagger
from nltk.translate.gleu_score import corpus_gleu
from transformers import AutoTokenizer, T5ForConditionalGeneration


tagger = Tagger('-Owakati')


def tokenize(sentence):
    return [t.surface for t in tagger(sentence)]


def main(model_dir, tokenizer_dir, corpus_dir, batch_size=64):
    source_path = glob.glob(os.path.join(corpus_dir, '*.src'))[0]
    with open(source_path, 'r', encoding='utf-8') as f:
        source_sents = [line for line in f if line]
    reference_tokens = []
    for reference_path in glob.glob(os.path.join(corpus_dir, '*.ref*')):
        with open(reference_path, 'r', encoding='utf-8') as f:
            tokens = [tokenize(line) for line in f if line]
            reference_tokens.append(tokens)
    reference_tokens = list(zip(*reference_tokens))
    print(f'Loaded {len(source_sents)} src, {len(reference_tokens)} ref')

    model = T5ForConditionalGeneration.from_pretrained(args.model_dir)
    model.config.max_length = 128
    model.cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)

    pred_tokens = []
    for i in range(0, len(source_sents), batch_size):
        input_ids = tokenizer(source_sents[i:i+batch_size], return_tensors='pt',
                              padding='max_length', max_length=128).input_ids
        input_ids = input_ids.to(device='cuda')
        gen_tokens = model.generate(input_ids)
        decoded = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
        pred_tokens.extend(tokenize(sent) for sent in decoded)
    print('Corpus GLEU', corpus_gleu(reference_tokens, pred_tokens))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_dir', required=True,
                        help='Path to pretrained model')
    parser.add_argument('-t', '--tokenizer_dir', required=True,
                        help='Path to pretrained tokenizer')
    parser.add_argument('-c', '--corpus_dir', required=True,
                        help='Path to directory of TMU evaluation corpus')
    args = parser.parse_args()
    main(args.model_dir, args.tokenizer_dir, args.corpus_dir)

import argparse
from pathlib import Path
import unicodedata

import h5py
from transformers import AutoTokenizer


def split_sentences(text):
    quote_lvl = 0
    brackets_lvl = 0
    start_i = 0
    sents = []
    for i, c in enumerate(text):
        if c == '「':
            quote_lvl += 1
        elif c == '」':
            quote_lvl -= 1
        elif c == '(':
            brackets_lvl += 1
        elif c == ')':
            brackets_lvl -= 1
        elif c in '。!?' and quote_lvl == 0 and brackets_lvl == 0:
            sents.append(text[start_i:i+1])
            start_i = i+1
    if start_i < len(text):
        sents.append(text[start_i:])
    res = []
    for sent in sents:
        sent = sent.lstrip('。').strip()
        if sent:
            res.append(sent)
    return res


def write_to_h5(tokens, path):
    L = len(tokens[0])
    with h5py.File(path, 'a') as f:
        if len(f) == 0:
            f.create_dataset('tgt', (0, L), maxshape=(None, L), dtype='i2')
        i = len(f['tgt'])
        n = i + len(tokens)
        f['tgt'].resize(n, axis=0)
        f['tgt'][i:n] = tokens


def main(args, batch_size=100000):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    current_file_i = 0
    current_file_len = 0

    def tokenize_and_write_to_h5(sents):
        tokens = tokenizer(sents, padding='max_length', truncation=True,
            max_length=args.max_length, return_tensors='np').input_ids
        current_file_path = Path(args.output_dir) / f'{current_file_i}.h5'
        write_to_h5(tokens, current_file_path)

    sents = []
    with open(args.corpus_path, 'r', encoding='utf-8') as cf:
        for line in cf:
            line = line.strip()
            if not line:
                continue
            line = unicodedata.normalize('NFKC', line)
            sents += split_sentences(line)
            if len(sents) >= batch_size:
                tokenize_and_write_to_h5(sents)
                current_file_len += len(sents)
                sents.clear()
                if current_file_len >= args.split_size:
                    current_file_i += 1
                    current_file_len = 0
    if sents:
        tokenize_and_write_to_h5(sents)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--corpus_path', required=True,
                        help='Corpus text file')
    parser.add_argument('-o', '--output_dir', required=True,
                        help='Split and tokenized corpus output directory')
    parser.add_argument('-s', '--split_size', type=int, required=True,
                        help='Number of sentences per split')
    parser.add_argument('-t', '--tokenizer_dir', required=True,
                        help='Pretrained tokenizer directory')
    parser.add_argument('-l', '--max_length', type=int, required=True,
                        help='Max sequence length for padding/truncation')
    args = parser.parse_args()
    main(args)

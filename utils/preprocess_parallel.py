import argparse

import h5py
from transformers import AutoTokenizer


def main(args, batch_size=128):
    with open(args.src_lines_path, encoding='utf-8') as f:
        src_sents = [l.strip() for l in f.readlines() if l.strip()]
    with open(args.tgt_lines_path, encoding='utf-8') as f:
        tgt_sents = [l.strip() for l in f.readlines() if l.strip()]
    n = len(src_sents)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)

    def tokenize_batch(sentences):
        return tokenizer(sentences, max_length=args.max_length, truncation=True,
            padding='max_length', return_tensors='np').input_ids

    with h5py.File(args.output_path, 'w') as f:
        f.create_dataset('src', (n, args.max_length), dtype='i2')
        f.create_dataset('tgt', (n, args.max_length), dtype='i2')
        for i in range(0, n, batch_size):
            f['src'][i:i+batch_size] = tokenize_batch(src_sents[i:i+batch_size])
            f['tgt'][i:i+batch_size] = tokenize_batch(tgt_sents[i:i+batch_size])
    print(f'Tokenized and saved {n} sentence pairs to {args.output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--src_lines_path', required=True,
                        help='Path to source lines in parallel corpus')
    parser.add_argument('-t', '--tgt_lines_path', required=True,
                        help='Path to target lines in parallel corpus')
    parser.add_argument('-d', '--tokenizer_dir', required=True,
                        help='Pretrained tokenizer directory')
    parser.add_argument('-o', '--output_path', required=True,
                        help='Path to tokenized corpus output')
    parser.add_argument('-l', '--max_length', type=int, required=True,
                        help='Max sequence length for padding/truncation')
    args = parser.parse_args()
    main(args)

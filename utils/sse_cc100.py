import argparse

import h5py
from hanziconv import HanziConv
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer


def sse(sentence, r=0.003):
    p = np.random.uniform(size=len(sentence))
    chars = [error_char(c) if p[i] <= r else c for i, c in enumerate(sentence)]
    sse_sent = ''.join(chars)
    return sse_sent


hira_range = (ord('ぁ'), ord('ん') + 1)
kata_range = (ord('ァ'), ord('ン') + 1)


def error_char(c):
    if hira_range[0] <= ord(c) < hira_range[1]:
        return chr(np.random.randint(*hira_range))
    elif kata_range[0] <= ord(c) < kata_range[1]:
        return chr(np.random.randint(*kata_range))
    else:
        return HanziConv.toSimplified(c)


def main(args, batch_size=100000):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)

    with h5py.File(args.corpus_path, 'a') as f:
        N, L = N, L = f['src'].shape
        for i in tqdm(range(0, N, batch_size)):
            tokens = f['src'][i:i+batch_size]
            sents = tokenizer.batch_decode(tokens, skip_special_tokens=True)
            sse_sents = [sse(sent) for sent in sents]
            sse_tokens = tokenizer(sse_sents, padding='max_length',
                truncation=True, max_length=L, return_tensors='np').input_ids
            f['src'][i:i+batch_size] = sse_tokens


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--corpus_path', required=True,
                        help='Tokenized h5 parallel corpus file')
    parser.add_argument('-t', '--tokenizer_dir', required=True,
                        help='Pretrained tokenizer directory')
    args = parser.parse_args()
    main(args)

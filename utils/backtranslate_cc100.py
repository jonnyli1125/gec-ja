import argparse

import h5py
import numpy as np
import torch
from tqdm import tqdm
from transformers import T5ForConditionalGeneration


class ScoreNoiser:
    def __init__(self, beta_random):
        self.beta_random = beta_random

    def __call__(self, scores, **kwargs):
        return self.beta_random * torch.rand_like(scores) + scores


def backtranslate(model, input_ids):
    input_ids = torch.tensor(input_ids, dtype=torch.long,
        device=torch.device('cuda:0'))
    return model.generate(input_ids, num_beams=8).cpu().numpy()


def main(args):
    model = T5ForConditionalGeneration.from_pretrained(args.model_dir)
    model.adjust_logits_during_generation = ScoreNoiser(args.beta_random)
    model.cuda()

    with h5py.File(args.corpus_path, 'a') as f:
        N, L = f['tgt'].shape
        model.config.max_length = L
        if 'src' not in f:
            f.create_dataset('src', (N, L), dtype='i2')
        B = args.batch_size
        if args.n_steps:
            range_end = min((args.i_offset + args.n_steps) * B, N)
        else:
            range_end = N
        for i in tqdm(range(args.i_offset * B, range_end, B)):
            input_ids = f['tgt'][i:i+B]
            bt_tokens = backtranslate(model, input_ids)
            if bt_tokens.shape[1] < L:
                bt_tokens = np.pad(bt_tokens, (0, L - bt_tokens.shape[1]))
            f['src'][i:i+B] = bt_tokens


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_dir', required=True,
                        help='Pretrained backtranslation model directory')
    parser.add_argument('-c', '--corpus_path', required=True,
                        help='Tokenized h5 corpus file')
    parser.add_argument('-br', '--beta_random', type=float, default=7,
                        help='Coefficient for noise in beam search scores')
    parser.add_argument('-b', '--batch_size', type=int, default=64,
                        help='Generation batch size')
    parser.add_argument('-s', '--n_steps', type=int,
                        help='Number of steps (batches) to generate')
    parser.add_argument('-i', '--i_offset', type=int, default=0,
                        help='Start generating from batch at index i')
    args = parser.parse_args()
    main(args)

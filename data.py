import h5py
import torch


class ParallelCorpusDataset(torch.utils.data.Dataset):
    def __init__(self, h5_path, inverse=False, pad_token_id=0):
        self.data = h5py.File(h5_path, 'r')
        if inverse:
            self.inputs_key = 'tgt'
            self.labels_key = 'src'
        else:
            self.inputs_key = 'src'
            self.labels_key = 'tgt'
        self.pad_token_id = pad_token_id

    def __len__(self):
        return len(self.data['src'])

    def __getitem__(self, idx):
        input_ids = torch.tensor(self.data[self.inputs_key][idx],
            dtype=torch.long)
        labels = torch.tensor(self.data[self.labels_key][idx], dtype=torch.long)
        return {
            'input_ids': input_ids,
            'attention_mask': input_ids != self.pad_token_id,
            'labels': labels
        }


def read_parallel_split(h5_path, inverse=False, val_ratio=0.01):
    dataset = ParallelCorpusDataset(h5_path, inverse)
    val_size = int(len(dataset) * val_ratio)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [len(dataset) - val_size, val_size])
    return train_dataset, val_dataset

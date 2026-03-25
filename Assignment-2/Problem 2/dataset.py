import torch
from torch.utils.data import Dataset

# special tokens every name sequence needs
SOS_TOKEN = '<SOS>'
EOS_TOKEN = '<EOS>'
PAD_TOKEN = '<PAD>'


class CharVocab:
    """char-to-index mapping built from the training names"""

    def __init__(self, names):
        chars = sorted(set(''.join(names)))
        self.special = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN]
        self.chars = self.special + chars

        self.ch2idx = {c: i for i, c in enumerate(self.chars)}
        self.idx2ch = {i: c for c, i in self.ch2idx.items()}

        self.pad_idx = self.ch2idx[PAD_TOKEN]
        self.sos_idx = self.ch2idx[SOS_TOKEN]
        self.eos_idx = self.ch2idx[EOS_TOKEN]

    def encode(self, name):
        return [self.ch2idx[c] for c in name]

    def decode(self, indices):
        # stop at EOS, skip SOS and PAD
        result = []
        for i in indices:
            ch = self.idx2ch[i]
            if ch == EOS_TOKEN:
                break
            if ch not in (SOS_TOKEN, PAD_TOKEN):
                result.append(ch)
        return ''.join(result)

    @property
    def size(self):
        return len(self.chars)


class NameDataset(Dataset):
    """converts names into (input, target) pairs for language model training"""

    def __init__(self, names, vocab):
        self.vocab = vocab
        self.data = []
        for name in names:
            encoded = vocab.encode(name)
            # input = SOS + chars, target = chars + EOS
            inp = [vocab.sos_idx] + encoded
            tgt = encoded + [vocab.eos_idx]
            self.data.append((torch.tensor(inp), torch.tensor(tgt)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch, pad_idx=0):
    """pad sequences to the same length so they can be stacked into a batch"""
    inputs, targets = zip(*batch)
    max_len = max(x.size(0) for x in inputs)

    padded_inp = torch.full((len(inputs), max_len), pad_idx, dtype=torch.long)
    padded_tgt = torch.full((len(targets), max_len), pad_idx, dtype=torch.long)

    for i, (inp, tgt) in enumerate(zip(inputs, targets)):
        padded_inp[i, :inp.size(0)] = inp
        padded_tgt[i, :tgt.size(0)] = tgt

    return padded_inp, padded_tgt


def load_names(path="TrainingNames.txt"):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

import torch
import os

from dataset import CharVocab
from models import VanillaRNN, BLSTM, RNNAttention

EMBED_DIM   = 32
HIDDEN_DIM  = 128
NUM_SAMPLES = 100
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def rebuild_vocab(data):
    """rebuild the vocab object from the saved checkpoint without re-loading training data"""
    vocab = CharVocab.__new__(CharVocab)
    vocab.chars   = data['vocab_chars']
    vocab.ch2idx  = {c: i for i, c in enumerate(vocab.chars)}
    vocab.idx2ch  = {i: c for c, i in vocab.ch2idx.items()}
    vocab.pad_idx = vocab.ch2idx['<PAD>']
    vocab.sos_idx = vocab.ch2idx['<SOS>']
    vocab.eos_idx = vocab.ch2idx['<EOS>']
    return vocab


def main():
    data  = torch.load('vocab_data.pt', weights_only=False)
    vocab = rebuild_vocab(data)

    models_config = [
        ("Vanilla RNN",    "vanilla_rnn",    VanillaRNN(vocab.size, EMBED_DIM, HIDDEN_DIM)),
        ("BLSTM",          "blstm",          BLSTM(vocab.size, EMBED_DIM, HIDDEN_DIM)),
        ("RNN + Attention","rnn_attention",  RNNAttention(vocab.size, EMBED_DIM, HIDDEN_DIM)),
    ]

    os.makedirs("generated_samples", exist_ok=True)

    for display_name, file_name, model in models_config:
        weight_path = f"saved_models/{file_name}.pt"
        if not os.path.exists(weight_path):
            print(f"Skipping {display_name} — weights not found at {weight_path}")
            continue

        model.load_state_dict(torch.load(weight_path, map_location=DEVICE, weights_only=True))
        model = model.to(DEVICE)

        print(f"\n{'='*50}")
        print(f"Generating {NUM_SAMPLES} names with {display_name}")
        print(f"{'='*50}")

        generated = []
        for _ in range(NUM_SAMPLES):
            name = model.generate(vocab, max_len=20, temperature=0.8)
            if name:
                generated.append(name)

        out_path = f"generated_samples/{file_name}_names.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            for n in generated:
                f.write(n + "\n")

        print(f"Generated {len(generated)} names → {out_path}")
        print("Samples:", ', '.join(generated[:15]))


if __name__ == "__main__":
    main()

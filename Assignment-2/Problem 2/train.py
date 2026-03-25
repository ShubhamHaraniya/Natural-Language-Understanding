import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import sys

from dataset import load_names, CharVocab, NameDataset, collate_fn
from models import VanillaRNN, BLSTM, RNNAttention

# training hyperparameters — easy to tweak from here
EMBED_DIM  = 32
HIDDEN_DIM = 128
LR         = 0.003
BATCH_SIZE = 64
EPOCHS     = 100
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_model(model, dataloader, vocab, epochs, name):
    print(f"\n{'='*60}")
    print(f"Training: {name}")
    print(f"{'='*60}")
    print(model.description())
    print(f"Trainable parameters: {count_params(model):,}")
    print(f"Hyperparameters: hidden={HIDDEN_DIM}, embed={EMBED_DIM}, lr={LR}, batch={BATCH_SIZE}, epochs={epochs}")
    print(f"Device: {DEVICE}\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)
    losses = []

    for epoch in range(1, epochs + 1):
        total_loss = 0
        batches    = 0

        for inp, tgt in dataloader:
            inp, tgt = inp.to(DEVICE), tgt.to(DEVICE)
            out = model(inp)

            # BLSTM returns (forward_logits, bidirectional_logits)
            # train both paths so the forward cell actually learns well
            if isinstance(out, tuple):
                fwd_out, bi_out = out
                loss = (criterion(fwd_out.reshape(-1, vocab.size), tgt.reshape(-1)) +
                        0.5 * criterion(bi_out.reshape(-1, vocab.size), tgt.reshape(-1)))
            else:
                loss = criterion(out.reshape(-1, vocab.size), tgt.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # clip so gradients don't explode
            optimizer.step()
            total_loss += loss.item()
            batches    += 1

        avg = total_loss / batches
        losses.append(avg)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs} | Loss: {avg:.4f}")

    return losses


def main():
    if not os.path.exists("TrainingNames.txt"):
        print("TrainingNames.txt not found. Run generate_names.py first.")
        sys.exit(1)

    names = load_names("TrainingNames.txt")
    print(f"Loaded {len(names)} training names")

    vocab = CharVocab(names)
    print(f"Vocabulary size: {vocab.size} ({vocab.size - 3} unique chars + 3 special tokens)")

    dataset = NameDataset(names, vocab)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                         collate_fn=lambda b: collate_fn(b, vocab.pad_idx))

    # save vocab so generate.py can load it without the full training set
    torch.save({'vocab_chars': vocab.chars, 'names': names}, 'vocab_data.pt')

    os.makedirs("saved_models", exist_ok=True)

    models_config = [
        ("Vanilla RNN",    VanillaRNN(vocab.size, EMBED_DIM, HIDDEN_DIM)),
        ("BLSTM",          BLSTM(vocab.size, EMBED_DIM, HIDDEN_DIM)),
        ("RNN + Attention", RNNAttention(vocab.size, EMBED_DIM, HIDDEN_DIM)),
    ]

    all_losses = {}

    for name, model in models_config:
        model  = model.to(DEVICE)
        losses = train_model(model, loader, vocab, EPOCHS, name)
        all_losses[name] = losses

        fname = name.lower().replace(' ', '_').replace('+', '').replace('__', '_')
        torch.save(model.state_dict(), f"saved_models/{fname}.pt")
        print(f"  Saved weights to saved_models/{fname}.pt")

    # plot all three loss curves together
    plt.figure(figsize=(10, 6))
    for name, losses in all_losses.items():
        plt.plot(losses, label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("training_loss.png", dpi=150)
    print(f"\nLoss curves saved to training_loss.png")


if __name__ == "__main__":
    main()

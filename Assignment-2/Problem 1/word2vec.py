# word2vec.py — CBOW and Skip-gram with Negative Sampling, built from scratch with NumPy

import os
import re
import numpy as np
from collections import Counter
import time


# three configs to compare — we pick the one with the lowest final loss
CONFIGS = [
    {"name": "config_A", "embed_dim": 100, "window": 5, "neg_samples": 5,  "lr": 0.025, "epochs": 50},
    {"name": "config_B", "embed_dim": 50,  "window": 3, "neg_samples": 5,  "lr": 0.025, "epochs": 50},
    {"name": "config_C", "embed_dim": 100, "window": 7, "neg_samples": 10, "lr": 0.025, "epochs": 50},
]
MIN_FREQ = 2  # drop words that appear less than this


def load_corpus(path="cleaned_corpus.txt"):
    # one sentence per line, already tokenized
    sentences = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) >= 2:
                sentences.append(tokens)
    return sentences


def build_vocab(sentences, min_freq=MIN_FREQ):
    # count and filter
    freq = Counter(word for sent in sentences for word in sent)
    vocab = sorted([w for w, c in freq.items() if c >= min_freq])
    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for w, i in word2idx.items()}
    return vocab, word2idx, idx2word, freq


def build_noise_dist(vocab, word2idx, freq):
    # unigram^0.75 — from the original Word2Vec paper
    # smooths rare words so they still get sampled as negatives
    freqs = np.array([freq.get(w, 0) ** 0.75 for w in vocab])
    return freqs / freqs.sum()


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))


def neg_sampling_loss_and_grad(center_vec, ctx_vec, neg_vecs):
    # positive pair should have high dot product, negatives should have low
    pos_score = sigmoid(ctx_vec @ center_vec)
    loss = -np.log(pos_score + 1e-10)
    grad_center = -(1 - pos_score) * ctx_vec
    grad_ctx = -(1 - pos_score) * center_vec

    grad_negs = []
    for neg_vec in neg_vecs:
        neg_score = sigmoid(-neg_vec @ center_vec)
        loss += -np.log(neg_score + 1e-10)
        grad_center += (1 - neg_score) * neg_vec
        grad_negs.append((1 - neg_score) * center_vec)

    return loss, grad_center, grad_ctx, grad_negs


def train_skipgram(sentences, vocab, word2idx, noise_dist, cfg):
    V = len(vocab)
    d = cfg["embed_dim"]
    window = cfg["window"]
    k = cfg["neg_samples"]
    lr = cfg["lr"]
    epochs = cfg["epochs"]

    W_in  = np.random.uniform(-0.5/d, 0.5/d, (V, d))  # center word matrix
    W_out = np.zeros((V, d))                             # context matrix

    vocab_idx = np.arange(V)
    total_pairs = sum(
        len(sent) * min(2 * window, len(sent) - 1)
        for sent in sentences
    )

    print(f"\n  Skip-gram {cfg['name']}: dim={d}, window={window}, neg={k}, epochs={epochs}")

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        # decay the learning rate linearly
        cur_lr = max(lr * (1 - (epoch - 1) / epochs), lr * 0.0001)

        np.random.shuffle(sentences)
        for sent in sentences:
            indices = [word2idx[w] for w in sent if w in word2idx]
            if len(indices) < 2:
                continue

            for pos, center_idx in enumerate(indices):
                ctx_range = range(max(0, pos - window), min(len(indices), pos + window + 1))
                for ctx_pos in ctx_range:
                    if ctx_pos == pos:
                        continue
                    ctx_idx = indices[ctx_pos]

                    neg_indices = np.random.choice(vocab_idx, size=k, replace=False, p=noise_dist)
                    neg_indices = [n for n in neg_indices if n != center_idx and n != ctx_idx][:k]
                    if len(neg_indices) < k:
                        continue

                    center_vec = W_in[center_idx]
                    ctx_vec    = W_out[ctx_idx]
                    neg_vecs   = [W_out[n] for n in neg_indices]

                    loss, g_center, g_ctx, g_negs = neg_sampling_loss_and_grad(
                        center_vec, ctx_vec, neg_vecs
                    )
                    total_loss += loss

                    W_in[center_idx] -= cur_lr * g_center
                    W_out[ctx_idx]   -= cur_lr * g_ctx
                    for i, n in enumerate(neg_indices):
                        W_out[n] -= cur_lr * g_negs[i]

        if epoch % 10 == 0 or epoch == 1:
            print(f"    Epoch {epoch:3d}/{epochs} | LR: {cur_lr:.5f} | Avg Loss: {total_loss / max(total_pairs, 1):.4f}")

    return W_in, total_loss / max(total_pairs, 1)


def train_cbow(sentences, vocab, word2idx, noise_dist, cfg):
    V = len(vocab)
    d = cfg["embed_dim"]
    window = cfg["window"]
    k = cfg["neg_samples"]
    lr = cfg["lr"]
    epochs = cfg["epochs"]

    W_in  = np.random.uniform(-0.5/d, 0.5/d, (V, d))
    W_out = np.zeros((V, d))

    vocab_idx = np.arange(V)

    print(f"\n  CBOW {cfg['name']}: dim={d}, window={window}, neg={k}, epochs={epochs}")

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        cur_lr = max(lr * (1 - (epoch - 1) / epochs), lr * 0.0001)

        np.random.shuffle(sentences)
        for sent in sentences:
            indices = [word2idx[w] for w in sent if w in word2idx]
            if len(indices) < 2:
                continue

            for pos, center_idx in enumerate(indices):
                ctx_indices = [
                    indices[i] for i in range(max(0, pos - window), min(len(indices), pos + window + 1))
                    if i != pos
                ]
                if not ctx_indices:
                    continue

                # average the context embeddings as the CBOW input
                ctx_avg = W_in[ctx_indices].mean(axis=0)

                center_vec = W_out[center_idx]
                neg_indices = np.random.choice(vocab_idx, size=k, replace=False, p=noise_dist)
                neg_indices = [n for n in neg_indices if n != center_idx][:k]
                if len(neg_indices) < k:
                    continue
                neg_vecs = [W_out[n] for n in neg_indices]

                loss, g_ctx_avg, g_center, g_negs = neg_sampling_loss_and_grad(
                    ctx_avg, center_vec, neg_vecs
                )
                total_loss += loss

                # spread gradient equally back to each context word
                g_per_ctx = g_ctx_avg / len(ctx_indices)
                for ci in ctx_indices:
                    W_in[ci] -= cur_lr * g_per_ctx
                W_out[center_idx] -= cur_lr * g_center
                for i, n in enumerate(neg_indices):
                    W_out[n] -= cur_lr * g_negs[i]

        if epoch % 10 == 0 or epoch == 1:
            print(f"    Epoch {epoch:3d}/{epochs} | LR: {cur_lr:.5f} | Avg Loss: {total_loss / max(len(sentences), 1):.4f}")

    return W_in, total_loss / max(len(sentences), 1)


def main():
    if not os.path.exists("cleaned_corpus.txt"):
        print("cleaned_corpus.txt not found. Run preprocess.py first.")
        return

    os.makedirs("embeddings", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    print("Loading corpus...")
    sentences = load_corpus("cleaned_corpus.txt")
    print(f"  Sentences loaded: {len(sentences)}")

    vocab, word2idx, idx2word, freq = build_vocab(sentences)
    print(f"  Vocabulary size : {len(vocab)} words")

    # save vocab so analyze.py and visualize.py can use it
    with open("embeddings/vocab.txt", "w", encoding="utf-8") as f:
        for w in vocab:
            f.write(w + "\n")
    print("  Vocab saved to embeddings/vocab.txt")

    noise_dist = build_noise_dist(vocab, word2idx, freq)

    results_summary = []

    print("\n" + "="*60)
    print("SKIP-GRAM TRAINING")
    print("="*60)
    sg_best_emb, sg_best_loss, sg_best_cfg = None, float("inf"), None
    for cfg in CONFIGS:
        t0 = time.time()
        emb, final_loss = train_skipgram(sentences, vocab, word2idx, noise_dist, cfg)
        elapsed = time.time() - t0
        results_summary.append({"model": "skipgram", **cfg, "final_loss": round(final_loss, 4), "time_s": round(elapsed)})
        print(f"    Done in {elapsed:.0f}s | Final loss: {final_loss:.4f}")
        if final_loss < sg_best_loss:
            sg_best_loss, sg_best_emb, sg_best_cfg = final_loss, emb, cfg["name"]

    print("\n" + "="*60)
    print("CBOW TRAINING")
    print("="*60)
    cb_best_emb, cb_best_loss, cb_best_cfg = None, float("inf"), None
    for cfg in CONFIGS:
        t0 = time.time()
        emb, final_loss = train_cbow(sentences, vocab, word2idx, noise_dist, cfg)
        elapsed = time.time() - t0
        results_summary.append({"model": "cbow", **cfg, "final_loss": round(final_loss, 4), "time_s": round(elapsed)})
        print(f"    Done in {elapsed:.0f}s | Final loss: {final_loss:.4f}")
        if final_loss < cb_best_loss:
            cb_best_loss, cb_best_emb, cb_best_cfg = final_loss, emb, cfg["name"]

    print(f"\n  Best Skip-gram config: {sg_best_cfg} (loss: {sg_best_loss:.4f})")
    print(f"  Best CBOW config:      {cb_best_cfg} (loss: {cb_best_loss:.4f})")

    # only save the best config for each model
    np.save("embeddings/skipgram_embeddings.npy", sg_best_emb)
    np.save("embeddings/cbow_embeddings.npy", cb_best_emb)
    print("\nFinal embeddings saved to embeddings/")

    with open("results/hyperparameter_results.txt", "w", encoding="utf-8") as f:
        f.write("=== Hyperparameter Experiment Results ===\n\n")
        f.write(f"{'Model':<12} {'Config':<12} {'Dim':>6} {'Window':>8} {'Neg':>6} {'Final Loss':>12} {'Time(s)':>10}\n")
        f.write("-" * 68 + "\n")
        for r in results_summary:
            f.write(f"{r['model']:<12} {r['name']:<12} {r['embed_dim']:>6} {r['window']:>8} {r['neg_samples']:>6} {r['final_loss']:>12} {r['time_s']:>10}\n")
        f.write(f"\nSaved Skip-gram: {sg_best_cfg} (loss={sg_best_loss:.4f})\n")
        f.write(f"Saved CBOW:      {cb_best_cfg} (loss={cb_best_loss:.4f})\n")
    print("Hyperparameter results saved to results/hyperparameter_results.txt")
    print("\nTraining complete.")


if __name__ == "__main__":
    main()

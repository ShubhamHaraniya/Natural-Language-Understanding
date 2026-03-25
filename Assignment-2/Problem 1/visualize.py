# visualize.py — generate t-SNE plots and cosine similarity heatmap

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# sklearn is just for t-SNE, not part of the model
from sklearn.manifold import TSNE
from collections import Counter


QUERY_WORDS = ["research", "student", "phd", "semester", "degree",
               "programme", "faculty", "course", "registration", "thesis", "examination"]

# words grouped by semantic category — colors help visually separate them in t-SNE
WORD_CLUSTERS = {
    "programmes":   ["btech", "mtech", "phd", "msc", "mba", "doctoral", "degree", "programme"],
    "evaluation":   ["examination", "grade", "cgpa", "sgpa", "marks", "result", "semester", "credit"],
    "admin/people": ["student", "faculty", "professor", "dean", "senate", "committee", "advisor"],
    "research":     ["research", "thesis", "publication", "project", "dissertation", "laboratory"],
    "courses":      ["course", "elective", "core", "curriculum", "syllabus", "lecture", "tutorial"],
}

CLUSTER_COLORS = {
    "programmes":   "#4C72B0",
    "evaluation":   "#C44E52",
    "admin/people": "#55A868",
    "research":     "#8B6914",
    "courses":      "#8172B2",
}


def load_embeddings(model_name):
    emb_path = f"embeddings/{model_name}_embeddings.npy"
    if not os.path.exists(emb_path):
        print(f"  {emb_path} not found")
        return None, None, None
    embeddings = np.load(emb_path)
    with open("embeddings/vocab.txt", "r", encoding="utf-8") as f:
        vocab = [line.strip() for line in f]
    word2idx = {w: i for i, w in enumerate(vocab)}
    return embeddings, vocab, word2idx


def cosine_similarity(a, b):
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / n) if n > 0 else 0.0


def get_tsne_data(embeddings, vocab, word2idx):
    # pick cluster words that are actually in the vocab
    selected_words = []
    word_to_cluster = {}

    for cluster, words in WORD_CLUSTERS.items():
        for w in words:
            if w in word2idx and w not in word_to_cluster:
                selected_words.append(w)
                word_to_cluster[w] = cluster

    # pad with frequent words if we have fewer than 30 points
    if len(selected_words) < 30:
        with open("cleaned_corpus.txt", "r", encoding="utf-8") as f:
            freq = Counter(w for line in f for w in line.strip().split())
        extra = [w for w, _ in freq.most_common(100)
                 if w in word2idx and w not in word_to_cluster and len(w) > 3]
        for w in extra[:max(0, 30 - len(selected_words))]:
            selected_words.append(w)
            word_to_cluster[w] = "other"

    vecs = np.array([embeddings[word2idx[w]] for w in selected_words])
    return selected_words, vecs, word_to_cluster


def plot_tsne(embeddings, vocab, word2idx, title, save_path, ax=None):
    selected_words, vecs, word_to_cluster = get_tsne_data(embeddings, vocab, word2idx)

    if len(vecs) < 5:
        print(f"  Not enough words for t-SNE ({len(vecs)}), skipping {save_path}")
        return

    perplexity = min(30, len(vecs) - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
    reduced = tsne.fit_transform(vecs)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(12, 8))

    color_map = {**CLUSTER_COLORS, "other": "#AAAAAA"}

    for i, word in enumerate(selected_words):
        cluster = word_to_cluster.get(word, "other")
        color = color_map[cluster]
        ax.scatter(reduced[i, 0], reduced[i, 1], c=color, s=60, alpha=0.85, zorder=2)
        ax.annotate(word, (reduced[i, 0], reduced[i, 1]),
                    fontsize=7.5, alpha=0.9, zorder=3,
                    xytext=(3, 3), textcoords="offset points")

    all_clusters = list(CLUSTER_COLORS.keys()) + ["other"]
    patches = [mpatches.Patch(color=color_map[c], label=c) for c in all_clusters if c in color_map]
    ax.legend(handles=patches, loc="best", fontsize=8, framealpha=0.8)
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.grid(True, alpha=0.15)

    if standalone:
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"  Saved: {save_path}")


def plot_similarity_heatmap(sg_emb, cb_emb, word2idx, save_path):
    words = [w for w in QUERY_WORDS if w in word2idx]
    if len(words) < 3:
        print(f"  Not enough query words in vocab for heatmap, skipping.")
        return

    def sim_matrix(emb):
        mat = np.zeros((len(words), len(words)))
        for i, w1 in enumerate(words):
            for j, w2 in enumerate(words):
                mat[i, j] = cosine_similarity(emb[word2idx[w1]], emb[word2idx[w2]])
        return mat

    sg_mat = sim_matrix(sg_emb)
    cb_mat = sim_matrix(cb_emb)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, mat, title in zip(axes, [sg_mat, cb_mat], ["Skip-gram", "CBOW"]):
        im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(words)))
        ax.set_yticks(range(len(words)))
        ax.set_xticklabels(words, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(words, fontsize=9)
        ax.set_title(f"Cosine Similarity Heatmap — {title}", fontsize=11)
        for i in range(len(words)):
            for j in range(len(words)):
                ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                        fontsize=7, color="white" if mat[i, j] > 0.6 else "black")
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def main():
    os.makedirs("plots", exist_ok=True)

    sg_emb, sg_vocab, sg_w2i = load_embeddings("skipgram")
    cb_emb, cb_vocab, cb_w2i = load_embeddings("cbow")

    if not os.path.exists("plots/wordcloud.png"):
        print("Word cloud not found — run preprocess.py first.")
    else:
        print("  plots/wordcloud.png already exists.")

    if sg_emb is not None:
        print("\nGenerating t-SNE for Skip-gram...")
        plot_tsne(sg_emb, sg_vocab, sg_w2i,
                  "t-SNE Projection — Skip-gram Embeddings",
                  "plots/tsne_skipgram.png")

    if cb_emb is not None:
        print("\nGenerating t-SNE for CBOW...")
        plot_tsne(cb_emb, cb_vocab, cb_w2i,
                  "t-SNE Projection — CBOW Embeddings",
                  "plots/tsne_cbow.png")

    if sg_emb is not None and cb_emb is not None:
        print("\nGenerating cosine similarity heatmap...")
        plot_similarity_heatmap(sg_emb, cb_emb, sg_w2i, "plots/top_neighbors_heatmap.png")

    print("\nAll plots saved to plots/")


if __name__ == "__main__":
    main()

# analyze.py — load trained embeddings and run nearest-neighbor + analogy tests

import numpy as np
import os


# words we want to explore
QUERY_WORDS = ["research", "student", "phd", "semester", "degree",
               "programme", "faculty", "course", "registration", "thesis", "examination"]

# format: (A, B, C, expected_answer)  →  vec(B) - vec(A) + vec(C) ≈ vec(expected)
ANALOGIES = [
    ("undergraduate", "btech",  "postgraduate", "mtech"),
    ("mtech",         "master", "btech",        "bachelor"),
    ("course",        "credit", "semester",     "cgpa"),
    ("student",       "grade",  "research",     "thesis"),
    ("science",       "msc",    "engineering",  "mtech"),
]


def load_embeddings(model_name):
    emb_path = f"embeddings/{model_name}_embeddings.npy"
    if not os.path.exists(emb_path):
        return None, None, None
    embeddings = np.load(emb_path)
    with open("embeddings/vocab.txt", "r", encoding="utf-8") as f:
        vocab = [line.strip() for line in f]
    word2idx = {w: i for i, w in enumerate(vocab)}
    return embeddings, vocab, word2idx


def cosine_sim(a, b):
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / n) if n > 0 else 0.0


def top_k(word, embeddings, word2idx, k=5):
    if word not in word2idx:
        return None
    q = embeddings[word2idx[word]]
    sims = [(w, cosine_sim(q, embeddings[i])) for w, i in word2idx.items() if w != word]
    return sorted(sims, key=lambda x: x[1], reverse=True)[:k]


def solve_analogy(a, b, c, embeddings, word2idx, k=10):
    # vec(B) - vec(A) + vec(C) — exclude the three query words from results
    missing = [w for w in [a, b, c] if w not in word2idx]
    if missing:
        return None, missing
    result_vec = (embeddings[word2idx[b]] - embeddings[word2idx[a]] + embeddings[word2idx[c]])
    exclude = {a, b, c}
    sims = [(w, cosine_sim(result_vec, embeddings[i])) for w, i in word2idx.items() if w not in exclude]
    return sorted(sims, key=lambda x: x[1], reverse=True)[:k], []


def run_analysis(model_name, label, log):
    separator = "=" * 65
    log.append(f"\n{separator}")
    log.append(f"  {label}")
    log.append(separator)

    embeddings, vocab, word2idx = load_embeddings(model_name)
    if embeddings is None:
        msg = f"  Could not load {model_name} embeddings — run word2vec.py first."
        print(msg); log.append(msg)
        return {}

    dim_line = f"  Loaded: {embeddings.shape[0]} words × {embeddings.shape[1]} dims\n"
    print(dim_line); log.append(dim_line)

    nn_section = f"{'─'*65}\n  TOP-5 NEAREST NEIGHBORS\n{'─'*65}"
    print(nn_section); log.append(nn_section)

    neighbor_results = {}
    for word in QUERY_WORDS:
        neighbors = top_k(word, embeddings, word2idx)
        if neighbors is None:
            line = f"  {word:<18} →  (not in vocabulary)"
        else:
            parts = [f"{w}  ({s:.3f})" for w, s in neighbors]
            line = f"  {word:<18} →  {',   '.join(parts)}"
            neighbor_results[word] = neighbors
        print(line); log.append(line)

    analogy_section = f"\n{'─'*65}\n  ANALOGY EXPERIMENTS\n  formula: vec(B) − vec(A) + vec(C)\n{'─'*65}"
    print(analogy_section); log.append(analogy_section)

    analogy_results = {}
    for a, b, c, expected in ANALOGIES:
        header = f"\n  {b} − {a} + {c}   →   '{expected}'"
        print(header); log.append(header)

        results, missing = solve_analogy(a, b, c, embeddings, word2idx)
        if results is None:
            msg = f"    Skipped — words not in vocab: {missing}"
            print(msg); log.append(msg)
            continue

        top_words = [w for w, _ in results]
        hit = expected in top_words

        for rank, (w, s) in enumerate(results, 1):
            flag = "   ← MATCH" if w == expected else ""
            line = f"    #{rank}  {w:<22}  {s:.3f}{flag}"
            print(line); log.append(line)

        verdict = f"    → '{expected}' {'✓ FOUND in top-10' if hit else '✗ not in top-10'}"
        print(verdict); log.append(verdict)
        analogy_results[(a, b, c, expected)] = (results, hit)

    return {"neighbors": neighbor_results, "analogies": analogy_results}


def main():
    os.makedirs("results", exist_ok=True)
    log = ["Word2Vec — Semantic Analysis Results", "=" * 65]

    run_analysis("skipgram", "Skip-gram + Negative Sampling", log)
    run_analysis("cbow",     "CBOW + Negative Sampling",      log)

    out_path = "results/analysis_results.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log))
    print(f"\n  Full results saved → {out_path}")


if __name__ == "__main__":
    main()

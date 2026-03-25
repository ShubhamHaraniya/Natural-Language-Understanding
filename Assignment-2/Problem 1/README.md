# Word Embeddings from IIT Jodhpur Data

Trains two Word2Vec models — **CBOW** and **Skip-gram with Negative Sampling** — completely from scratch (pure NumPy, no Gensim) on textual data collected from IIT Jodhpur's official sources. Then uses those embeddings to explore semantic structure through nearest-neighbor search and analogy experiments.

---

## What this project does

The models read through IIT Jodhpur's academic documents and learn vector representations of words based on how they appear together. Once trained, words with similar meanings end up pointing in similar directions in the embedding space. So querying "nearest neighbors of `phd`" should return things like `research`, `thesis`, `supervisor` — and it does.

---

## Project Structure

```
Problem 1/
│
├── corpus/                          # raw input files
│   ├── iitj_academic_regulations.txt
│   ├── iitj_academic_programs_list.txt
│   ├── iitj_cse_programs.txt
│   └── iitj_mtech_btech_curriculum.txt
│
├── preprocess.py                    # cleaning, stats, word cloud
├── word2vec.py                      # CBOW + Skip-gram from scratch
├── analyze.py                       # cosine similarity + 5 analogies
├── visualize.py                     # 5 plots
│
├── cleaned_corpus.txt               # one sentence per line
├── tokens.txt                       # flat token list
│
├── embeddings/
│   ├── vocab.txt                    # word list (one per line)
│   ├── cbow_embeddings.npy          # final trained CBOW vectors
│   └── skipgram_embeddings.npy      # final trained Skip-gram vectors
│
├── results/
│   ├── dataset_stats.txt            # total docs, tokens, vocab size
│   ├── hyperparameter_results.txt   # comparison of 3 configurations
│   └── analysis_results.txt         # nearest neighbors + analogy output
│
├── plots/
│   ├── wordcloud.png                # most frequent words
│   ├── tsne_skipgram.png            # t-SNE of Skip-gram embeddings
│   ├── tsne_cbow.png                # t-SNE of CBOW embeddings
│   └── top_neighbors_heatmap.png   # cosine similarity heatmap
│
├── README.md
└── report.tex
```

---

## How to run

> Requires: `numpy`, `matplotlib`, `scikit-learn`, `wordcloud`

```bash
# Step 1: clean the corpus and generate word cloud
python preprocess.py

# Step 2: train CBOW and Skip-gram (takes a few minutes)
python word2vec.py

# Step 3: analyze semantic structure
python analyze.py

# Step 4: generate all 4 plots
python visualize.py
```

---

## The two models

Both are built **from scratch** using NumPy — no Gensim, no PyTorch.

### CBOW (Continuous Bag of Words)
Takes the surrounding context words, averages their embeddings, and predicts the center word. Works well for frequent words and smooth embeddings.

### Skip-gram with Negative Sampling
Takes the center word and predicts each surrounding context word. Uses k noise words (negatives) per positive pair to avoid a full softmax over the vocabulary. Better at capturing rare word semantics.

| Model | Config | Embed dim | Window | Neg samples | Final Loss |
|---|---|---|---|---|---|
| Skip-gram | A | 100 | 5 | 5 | 1.1826 |
| Skip-gram | B (saved) | 50 | 3 | 5 | 1.1563 |
| Skip-gram | C | 100 | 7 | 10 | 1.4246 |
| CBOW | A | 100 | 5 | 5 | 24.5254 |
| CBOW | B (saved) | 50 | 3 | 5 | 21.9324 |
| CBOW | C | 100 | 7 | 10 | 29.8253 |

Only Config B's final weights are saved based on lowest final loss.

---

## Analogy experiments

Five analogy tests using `vec(B) - vec(A) + vec(C) → D`:

| A | B | C | Expected |
|---|---|---|---|
| student | grade | research | thesis |
| undergraduate | btech | postgraduate | mtech |
| mtech | master | btech | bachelor |
| course | credit | semester | cgpa |
| science | msc | engineering | mtech |

---

## Key finding

CBOW and Skip-gram both capture domain-specific academic structure. Skip-gram tends to be better at rare terms (like `phd`, `thesis`), while CBOW produces smoother representations for frequent terms like `student`, `course`. The analogy experiments work best for well-represented word pairs in the corpus.

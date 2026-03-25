# preprocess.py — clean the raw corpus and compute basic stats

import os
import re
from collections import Counter

CORPUS_DIR = "corpus"
CORPUS_FILES = [
    "iitj_academic_regulations.txt",
    "iitj_academic_programs_list.txt",
    "iitj_cse_programs.txt",
    "iitj_mtech_btech_curriculum.txt",
]

# common words that don't carry meaning — skip these in word cloud / domain stats
STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "in", "to", "for", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "will", "would", "could", "should", "may", "might", "shall", "can",
    "this", "that", "these", "those", "it", "its", "with", "as", "by", "from",
    "at", "on", "not", "no", "but", "if", "than", "so", "up", "out", "about",
    "into", "which", "who", "what", "when", "where", "how", "all", "any",
    "their", "they", "them", "he", "she", "his", "her", "we", "our", "us",
    "you", "your", "per", "also", "only", "both", "such", "each", "after",
    "before", "more", "other", "same"
}

def load_raw_text():
    docs = []
    for fname in CORPUS_FILES:
        path = os.path.join(CORPUS_DIR, fname)
        if not os.path.exists(path):
            print(f"  Warning: {fname} not found.")
            continue
        with open(path, "r", encoding="utf-8") as f:
            docs.append(f.read())
        print(f"  Loaded {fname} ({os.path.getsize(path) // 1024} KB)")
    return docs

def preprocess_document(doc):
    # drop non-ASCII garbage first
    doc = doc.encode("ascii", errors="ignore").decode("ascii")

    # strip URLs, emails, separator lines
    doc = re.sub(r"http\S+|www\.\S+|\S+@\S+", " ", doc)
    doc = re.sub(r"[\=\-\|\*\#]{4,}", " ", doc)

    # split into sentences — keep 'B.Tech.' intact by splitting on '. ' not '.'
    raw_sentences = re.split(r'\.\s+|\?\s+|\!\s+|\n+', doc)

    cleaned_sentences = []
    for sent in raw_sentences:
        sent = sent.strip()
        if not sent:
            continue

        tokens = sent.split()
        cleaned_tokens = []
        for t in tokens:
            t = t.lower()
            t = re.sub(r'[^a-z0-9]', '', t)
            # need at least 2 chars and one letter to be useful
            if len(t) >= 2 and re.search(r'[a-z]', t):
                cleaned_tokens.append(t)

        if len(cleaned_tokens) >= 3:
            cleaned_sentences.append(cleaned_tokens)

    return cleaned_sentences

def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    print("Loading corpus files...")
    docs = load_raw_text()

    print("\nPreprocessing...")
    all_sentences = []
    for doc in docs:
        all_sentences.extend(preprocess_document(doc))

    all_tokens = [t for sent in all_sentences for t in sent]
    freq = Counter(all_tokens)

    # keep only words seen at least twice
    vocab = [w for w, c in freq.items() if c >= 2]

    print(f"\nDataset Statistics:")
    print(f"  Source documents : {len(docs)}")
    print(f"  Sentences        : {len(all_sentences)}")
    print(f"  Total tokens     : {len(all_tokens)}")
    print(f"  Vocabulary size  : {len(vocab)} (words appearing >= 2 times)")

    domain_freq = {w: c for w, c in freq.items() if w not in STOPWORDS}
    top_domain = [(w, c) for w, c in Counter(domain_freq).most_common(20)]
    print(f"  Top 20 domain words: {top_domain}")

    # write cleaned sentences (one per line) and flat token list
    with open("cleaned_corpus.txt", "w", encoding="utf-8") as f:
        for sent in all_sentences:
            f.write(" ".join(sent) + "\n")

    with open("tokens.txt", "w", encoding="utf-8") as f:
        f.write(" ".join(all_tokens))

    with open("results/dataset_stats.txt", "w", encoding="utf-8") as f:
        f.write("=== Dataset Statistics ===\n\n")
        f.write(f"Source documents : {len(docs)}\n")
        f.write(f"Sentences        : {len(all_sentences)}\n")
        f.write(f"Total tokens     : {len(all_tokens)}\n")
        f.write(f"Vocabulary size  : {len(vocab)} (min freq = 2)\n\n")
        f.write("Top 50 domain words:\n")
        for word, count in Counter(domain_freq).most_common(50):
            f.write(f"  {word:<28} {count}\n")
    print("Stats saved to results/dataset_stats.txt")

    # word cloud — skip quietly if wordcloud not installed
    try:
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt

        wc = WordCloud(
            width=1400, height=600,
            background_color="white", colormap="Blues",
            max_words=150, prefer_horizontal=0.8,
        ).generate_from_frequencies(domain_freq)

        plt.figure(figsize=(16, 7))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title("Word Cloud — IIT Jodhpur Corpus", fontsize=16, pad=12)
        plt.tight_layout()
        plt.savefig("plots/wordcloud.png", dpi=150)
        plt.close()
        print("Word cloud saved to plots/wordcloud.png")
    except ImportError:
        pass

    print("\nPreprocessing done.")

if __name__ == "__main__":
    main()

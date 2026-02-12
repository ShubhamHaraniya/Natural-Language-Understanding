"""
NLP Assignment 1 - Problem 2
Student ID: M25CSA013
BPE Tokenization implementation
"""

import sys
import re
from collections import defaultdict, Counter

# using this to mark end of word
EOW = '</w>'

def get_data(filename):
    # simple file reading
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        return [l.strip() for l in lines if l.strip()]
    except:
        print("Couldn't read file")
        sys.exit(1)

def get_tokens(text):
    # split by whitespace and lowercase
    # removing punctuation might be good but keeping it simple
    return text.lower().split()

def build_vocab(lines):
    # counting words, adding end token to chars
    # vocab maps word (tuple of chars) -> frequency
    vocab = Counter()
    for line in lines:
        words = get_tokens(line)
        for w in words:
            # make tuple of chars: 'hi' -> ('h', 'i', '</w>')
            w_tuple = tuple(list(w) + [EOW])
            vocab[w_tuple] += 1
    return vocab

def get_stats(vocab):
    # find all adjacent pairs and their counts
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i+1])
            pairs[pair] += freq
    return pairs

def merge_vocab(pair, vocab):
    # taking the best pair and merging it in all words
    v_out = {}
    new_token = pair[0] + pair[1]
    
    for word, freq in vocab.items():
        new_word = []
        i = 0
        while i < len(word):
            # check if we found the pair
            if i < len(word) - 1 and word[i] == pair[0] and word[i+1] == pair[1]:
                new_word.append(new_token)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        v_out[tuple(new_word)] = freq
    return v_out

def main():
    if len(sys.argv) < 3:
        print("Usage: python M25CSA013_prob2.py <k> <corpus_file>")
        print("Example: python M25CSA013_prob2.py 50 corpus.txt")
        return

    k = int(sys.argv[1])
    filepath = sys.argv[2]
    
    print("Reading file " + filepath)
    data = get_data(filepath)
    
    # get initial counts
    vocab = build_vocab(data)
    
    print("Starting BPE with {} merges...".format(k))
    print("-" * 30)
    
    merges = []
    
    for i in range(k):
        pairs = get_stats(vocab)
        if not pairs:
            break
            
        # find max freq pair
        best = max(pairs, key=pairs.get)
        freq = pairs[best]
        
        # update our vocab with the merge
        vocab = merge_vocab(best, vocab)
        merges.append((best, freq))
        
        print("Merge {}: {} + {} -> {} (freq: {})".format(i+1, best[0], best[1], best[0]+best[1], freq))
    
    print("-" * 30)
    
    # printing final results
    print("\nFinal Vocabulary:")
    unique_tokens = set()
    for word in vocab.keys():
        for char in word:
            unique_tokens.add(char)
            
    sorted_tokens = sorted(list(unique_tokens), key=lambda x: (len(x), x))
    print(sorted_tokens)
    print("\nTotal vocabulary size: {}".format(len(unique_tokens)))

if __name__ == '__main__':
    main()
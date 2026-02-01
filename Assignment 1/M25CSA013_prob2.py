"""
# NLP Assignment 1 - Problem 2: BPE Tokenization
# implementing byte pair encoding from scratch as requested
# student id: M25CSA013
# 
# this script creates a bpe tokenizer without using any nlp libraries.
# the basic idea is:
# 1. start with char level tokens
# 2. find most frequent pair of adjacent tokens
# 3. merge them into a new token
# 4. repeat K times
# 
# usage: python M25CSA013_prob2.py corpus.txt <num_merges>
"""

import sys
import re
from collections import defaultdict, Counter

# separating this so i can change it easily if needed
# used to mark end of words for BPE
END_OF_WORD = '</w>'


def read_corpus(filepath):
    # helper function to just read the file and get lines
    # returns list of strings, filtering empty ones
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            # strip whitespace and skip empty lines
            lines = [line.strip() for line in file]
            lines = [line for line in lines if line]
            return lines
    except FileNotFoundError:
        print("Error: File '{}' not found.".format(filepath))
        sys.exit(1)
    except IOError as e:
        print("Error reading file: {}".format(e))
        sys.exit(1)


def tokenize_words(text):
    # just basic splitting. lowercasing everything to keep it simple.
    # regex matches any non-whitespace sequence.
    words = re.findall(r'\S+', text.lower())
    return words


def get_word_frequencies(corpus_lines):
    # we need to count how many times each word appears
    # but initially, every word is just a list of chars + end tag
    # e.g. "hi" -> ('h', 'i', '</w>')
    word_freq = Counter()
    
    for line in corpus_lines:
        # Tokenize the line into words
        words = tokenize_words(line)
        
        for word in words:
            # skip empty stuff
            if not word:
                continue
            
            # converting word string to char tuple with end marker
            # cat -> ('c', 'a', 't', '</w>')
            char_tuple = tuple(list(word) + [END_OF_WORD])
            word_freq[char_tuple] += 1
    
    return word_freq


def get_pair_frequencies(word_frequencies):
    # we need to check every adjacent pair of characters/tokens
    # and count how many times they appear together
    pair_counts = Counter()
    
    for word_tuple, freq in word_frequencies.items():
        # going through the word tuple to find pairs
        for i in range(len(word_tuple) - 1):
            pair = (word_tuple[i], word_tuple[i + 1])
            pair_counts[pair] += freq
    
    return pair_counts


def merge_pair(word_frequencies, pair_to_merge):
    # this replaces the frequent pair we found with a merged token
    # effectively updating our "vocab" in the corpus
    new_word_freq = {}
    
    # combined string
    merged_token = pair_to_merge[0] + pair_to_merge[1]
    
    for word_tuple, freq in word_frequencies.items():
        new_word = []
        i = 0
        
        while i < len(word_tuple):
            # checking if current and next token match our pair
            if i < len(word_tuple) - 1 and \
               word_tuple[i] == pair_to_merge[0] and \
               word_tuple[i + 1] == pair_to_merge[1]:
                # found match, put merged token
                new_word.append(merged_token)
                i += 2
            else:
                # no match, keep distinct
                new_word.append(word_tuple[i])
                i += 1
        
        new_word_freq[tuple(new_word)] = freq
    
    return new_word_freq


def get_initial_vocabulary(word_frequencies):
    # grabbing all unique chars to start our vocab
    vocab = set()
    
    for word_tuple in word_frequencies.keys():
        for token in word_tuple:
            vocab.add(token)
    
    return vocab


def run_bpe(corpus_lines, num_merges):
    # main loop for bpe
    # 1. init vocab with chars
    # 2. repeat K times: find best pair, merge it, update vocab
    print("\n" + "=" * 60)
    print("BPE TOKENIZATION PROCESS")
    print("=" * 60)
    
    # Step 1: get base word counts with char split
    print("\n[Step 1] Building word frequencies from corpus...")
    word_freq = get_word_frequencies(corpus_lines)
    print("  Found {} unique words in corpus".format(len(word_freq)))
    
    # Step 2: intialize vocab with all known chars
    vocabulary = get_initial_vocabulary(word_freq)
    print("\n[Step 2] Initial vocabulary size: {}".format(len(vocabulary)))
    print("  Initial tokens: {}".format(sorted(vocabulary)))
    
    # keeping track of what we merged
    merge_operations = []
    
    # Step 3: Perform K merge operations
    print("\n[Step 3] Performing {} merge operations...".format(num_merges))
    print("-" * 60)
    
    for i in range(num_merges):
        # get counts of all pairs
        pair_freqs = get_pair_frequencies(word_freq)
        
        # stop if nothing left to merge
        if not pair_freqs:
            print("\n  No more pairs to merge after {} operations.".format(i))
            break
        
        # find the pair with highest count
        best_pair = pair_freqs.most_common(1)[0]
        pair, freq = best_pair
        
        # make the new token
        merged_token = pair[0] + pair[1]
        
        # log it
        print("  Merge {}: '{}' + '{}' -> '{}' (freq: {})".format(
            i + 1, pair[0], pair[1], merged_token, freq))
        
        # actually do the merge in our word dict
        word_freq = merge_pair(word_freq, pair)
        
        # add to our vocab set
        vocabulary.add(merged_token)
        
        # save to history
        merge_operations.append((pair, freq, merged_token))
    
    print("-" * 60)
    print("\n[Step 4] Final vocabulary size: {}".format(len(vocabulary)))
    
    return vocabulary, merge_operations, word_freq


def display_vocabulary(vocabulary, merge_ops):
    # printing the final vocabulary
    print("\n" + "=" * 60)
    print("FINAL VOCABULARY")
    print("=" * 60)
    
    # separating basic chars from merged ones
    single_chars = []
    merged_tokens = []
    
    for token in vocabulary:
        if token == END_OF_WORD:
            continue 
        elif len(token) == 1:
            single_chars.append(token)
        else:
            merged_tokens.append(token)
    
    # sorting for display
    single_chars.sort()
    merged_tokens.sort(key=lambda x: (len(x), x))
    
    print("\n1. Single Characters ({} tokens):".format(len(single_chars)))
    print("   {}".format(single_chars))
    
    print("\n2. End-of-Word Marker:")
    print("   ['{}']".format(END_OF_WORD))
    
    print("\n3. Merged Tokens ({} tokens):".format(len(merged_tokens)))

    for i in range(0, len(merged_tokens), 10):
        row = merged_tokens[i:i+10]
        print("   {}".format(row))
    
    print("\n" + "=" * 60)
    print("TOTAL VOCABULARY SIZE: {} tokens".format(len(vocabulary)))
    print("  - Base characters: {}".format(len(single_chars)))
    print("  - End marker: 1")
    print("  - Merged tokens: {}".format(len(merged_tokens)))
    print("=" * 60)


def display_merge_history(merge_operations):
    # printing the table of merges
    print("\n" + "=" * 60)
    print("MERGE HISTORY")
    print("=" * 60)
    header = "{:<5} {:<15} {:<15} {:<15} {:<8}".format(
        "#", "Token 1", "Token 2", "Merged", "Freq")
    print(header)
    print("-" * 60)
    
    for i, (pair, freq, merged) in enumerate(merge_operations, 1):

        t1 = "'" + pair[0] + "'"
        t2 = "'" + pair[1] + "'"
        m = "'" + merged + "'"
        row = "{:<5} {:<15} {:<15} {:<15} {:<8}".format(i, t1, t2, m, freq)
        print(row)
    
    print("=" * 60)

def main():

    print("\n" + "=" * 70)
    print("  NLP Assignment 1 - Problem 2: BPE Tokenization")
    print("  Student ID: M25CSA013")
    print("=" * 70)
    
    if len(sys.argv) < 2:
        print("Usage: python M25CSA013_prob2.py <corpus_file> <num_merges>")
        sys.exit(1)
    
    # Parse arguments
    corpus_file = sys.argv[1]
    
    # default K if not provided
    num_merges = 50
    if len(sys.argv) > 2:
        try:
            num_merges = int(sys.argv[2])
            if num_merges < 0:
                raise ValueError("Number of merges must be non-negative")
        except ValueError as e:
            print("\nError: Invalid number of merges: {}".format(sys.argv[2]))
            print("Please provide a non-negative integer.")
            sys.exit(1)
    else:
        print("\nNote: No K provided. Using default K=50.")
    
    print("\nInput file: {}".format(corpus_file))
    print("Number of merges (K): {}".format(num_merges))
    
    # Read the corpus
    corpus_lines = read_corpus(corpus_file)
    print("Loaded {} lines from corpus".format(len(corpus_lines)))
    
    # Preview first few lines
    print("\nCorpus preview (first 3 lines):")
    for i, line in enumerate(corpus_lines[:3]):
        preview = line[:50] + "..." if len(line) > 50 else line
        print("  {}. {}".format(i + 1, preview))
    
    # Run BPE
    vocabulary, merge_ops, final_word_freq = run_bpe(corpus_lines, num_merges)
    
    # Display results
    display_merge_history(merge_ops)
    display_vocabulary(vocabulary, merge_ops)
    
    print("\nBPE tokenization completed successfully!")
    print("=" * 70 + "\n")
    
    return vocabulary

if __name__ == "__main__":
    main()  
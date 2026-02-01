"""
# NLP Assignment 1 - Problem 3: Naive Bayes Sentiment Classifier
# implementing standard naive bayes from scratch
# student id: M25CSA013
#
# this script creates a sentiment classifier using only standard libraries.
# the basic idea is:
# 1. read positive and negative files
# 2. split into train/val
# 3. train naive bayes with laplace smoothing
# 4. predict sentiment of new sentences
#
# usage: python M25CSA013_prob3.py
"""

import math
import random
import sys

# setting random seed so we get same results every time
random.seed(42)

def load_data(filename):
    # helper to read file line by line
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        sys.exit(1)

def tokenize(sentence):
    # basic tokenization: lowercasing and splitting by spaces
    return sentence.lower().split()

def split_data(data, split_ratio=0.8):
    # shuffling and splitting the data for training and validation
    random.shuffle(data)
    split_index = int(len(data) * split_ratio)
    return data[:split_index], data[split_index:]

def train_naive_bayes(pos_train, neg_train):
    """
    Trains a Naive Bayes classifier.
    Returns:
        log_priors: dict {'POS': log_prob, 'NEG': log_prob}
        log_likelihoods: dict {'POS': {word: log_prob}, 'NEG': {word: log_prob}}
        vocab: set of all words in training data
    """
    # 1. Calculate Class Priors
    num_pos = len(pos_train)
    num_neg = len(neg_train)
    total_docs = num_pos + num_neg
    
    # calculating prior probabilities (log prob)
    log_priors = {
        'POS': math.log(num_pos / total_docs),
        'NEG': math.log(num_neg / total_docs)
    }

    # 2. Build Vocabulary and Count Words per Class
    pos_word_counts = {}
    neg_word_counts = {}
    vocab = set()

    # going through positive docs to count words
    total_words_pos = 0
    for doc in pos_train:
        words = tokenize(doc)
        for w in words:
            vocab.add(w)
            pos_word_counts[w] = pos_word_counts.get(w, 0) + 1
            total_words_pos += 1

    # going through negative docs to count words
    total_words_neg = 0
    for doc in neg_train:
        words = tokenize(doc)
        for w in words:
            vocab.add(w)
            neg_word_counts[w] = neg_word_counts.get(w, 0) + 1
            total_words_neg += 1

    # 3. Calculate Likelihoods with Laplace Smoothing
    vocab_size = len(vocab)
    log_likelihoods = {'POS': {}, 'NEG': {}}

    # we need P(w|C) for every word in the vocabulary
    # using laplace smoothing: (count(w, C) + 1) / (count(all_words, C) + |V|)
    
    # pre-calculating the denominator parts (using logs to avoid underflow issues)
    denom_pos = math.log(total_words_pos + vocab_size)
    denom_neg = math.log(total_words_neg + vocab_size)

    for w in vocab:
        # for positive class
        count_w_pos = pos_word_counts.get(w, 0)
        log_likelihoods['POS'][w] = math.log(count_w_pos + 1) - denom_pos

        # for negative class
        count_w_neg = neg_word_counts.get(w, 0)
        log_likelihoods['NEG'][w] = math.log(count_w_neg + 1) - denom_neg
        
    # ignoring OOV words for now since they don't affect the relative scores
    # in a strict naive bayes sense, we only care about words we've seen before
    
    return log_priors, log_likelihoods, vocab

def predict(text, log_priors, log_likelihoods, vocab):
    """
    Predicts sentiment for a given text.
    Returns 'POSITIVE' or 'NEGATIVE'.
    """
    words = tokenize(text)
    
    # Initialize scores with priors
    score_pos = log_priors['POS']
    score_neg = log_priors['NEG']
    
    for w in words:
        if w in vocab:
            score_pos += log_likelihoods['POS'][w]
            score_neg += log_likelihoods['NEG'][w]
        # if word is not in vocab, just ignore it
            
    if score_pos > score_neg:
        return 'POSITIVE'
    else:
        return 'NEGATIVE'

def main():
    print("Loading data...")
    pos_data = load_data('pos.txt')
    neg_data = load_data('neg.txt')
    
    print(f"Loaded {len(pos_data)} positive and {len(neg_data)} negative sentences.")
    
    # splitting into train and test
    pos_train, pos_val = split_data(pos_data)
    neg_train, neg_val = split_data(neg_data)
    
    # training the model
    print("Training Naive Bayes Model...")
    log_priors, log_likelihoods, vocab = train_naive_bayes(pos_train, neg_train)
    
    # checking accuracy on the validation set
    print("Validating model...")
    correct = 0
    total = 0
    
    for doc in pos_val:
        if predict(doc, log_priors, log_likelihoods, vocab) == 'POSITIVE':
            correct += 1
        total += 1
    
    for doc in neg_val:
        if predict(doc, log_priors, log_likelihoods, vocab) == 'NEGATIVE':
            correct += 1
        total += 1
        
    accuracy = (correct / total) * 100 if total > 0 else 0
    print(f"Validation Accuracy: {accuracy:.2f}%")
    print("-" * 30)
    print("Enter a sentence to predict its sentiment (or type 'exit' to quit).")
    
    while True:
        try:
            user_input = input("Enter sentence: ")
            if user_input.lower() in ('exit', 'quit'):
                break
            
            sentiment = predict(user_input, log_priors, log_likelihoods, vocab)
            print(f"Prediction: {sentiment}")
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except EOFError:
             break

if __name__ == "__main__":
    main()
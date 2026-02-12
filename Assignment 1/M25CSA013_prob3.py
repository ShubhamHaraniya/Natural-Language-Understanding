"""
NLP Assignment 1 - Problem 3
Student ID: M25CSA013
Naive Bayes Classifier
"""

import math
import random
import sys


def load_file_data(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as file_obj:
        for raw_line in file_obj:
            clean_line = raw_line.strip()
            if clean_line:
                data.append(clean_line)
    return data

def tokenize_input(raw_text):
    # Converting to lowercase and splitting by whitespace
    return raw_text.lower().split()

def build_counts(documents):
    # count occurrences of each word and total word count
    counts = {}
    word_count = 0
    for text in documents:
        tokens = tokenize_input(text)
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
            word_count += 1
    return counts, word_count

def train_naive_bayes(positive_data, negative_data):
    # priors
    count_pos = len(positive_data)
    count_neg = len(negative_data)
    total_docs = count_pos + count_neg
    
    # log probabilities for classes
    log_prior_pos = math.log(count_pos / total_docs)
    log_prior_neg = math.log(count_neg / total_docs)
    
    # We need a set of all unique words across both classes
    unique_vocab = set()
    pos_freqs, pos_total_terms = build_counts(positive_data)
    neg_freqs, neg_total_terms = build_counts(negative_data)
    
    # update vocab set
    unique_vocab.update(pos_freqs.keys())
    unique_vocab.update(neg_freqs.keys())
            
    # likelihoods with smoothing
    model_pos_probs = {}
    model_neg_probs = {}
    
    vocab_size = len(unique_vocab)
    denominator_pos = pos_total_terms + vocab_size
    denominator_neg = neg_total_terms + vocab_size
    
    for word in unique_vocab:
        # Add-one smoothing
        raw_count_pos = pos_freqs.get(word, 0)
        p_word_pos = (raw_count_pos + 1) / denominator_pos
        model_pos_probs[word] = math.log(p_word_pos)
        
        raw_count_neg = neg_freqs.get(word, 0)
        p_word_neg = (raw_count_neg + 1) / denominator_neg
        model_neg_probs[word] = math.log(p_word_neg)
        
    return (log_prior_pos, log_prior_neg, model_pos_probs, model_neg_probs, unique_vocab)

def classify_text(input_text, trained_model):
    p_pos, p_neg, prob_map_pos, prob_map_neg, vocabulary = trained_model
    
    current_pos_score = p_pos
    current_neg_score = p_neg
    
    # update scores based on words in input
    clean_words = tokenize_input(input_text)
    
    for token in clean_words:
        if token in vocabulary:
            current_pos_score += prob_map_pos[token]
            current_neg_score += prob_map_neg[token]
            
    if current_pos_score > current_neg_score:
        return 'POSITIVE'
    else:
        return 'NEGATIVE'

def run_classifier():
    # Set seed for consistent results
    random.seed(42)

    list_pos = load_file_data('pos.txt')
    list_neg = load_file_data('neg.txt')
    
    # Shuffle in place
    random.shuffle(list_pos)
    random.shuffle(list_neg)
    
    # 80/20 train test split
    idx_p = int(len(list_pos) * 0.8)
    idx_n = int(len(list_neg) * 0.8)
    
    training_set_pos = list_pos[:idx_p]
    testing_set_pos = list_pos[idx_p:]
    
    training_set_neg = list_neg[:idx_n]
    testing_set_neg = list_neg[idx_n:]
    
    total_train = len(training_set_pos) + len(training_set_neg)
    print(f"Training on {total_train} docs")
    
    # Train model
    nb_model = train_naive_bayes(training_set_pos, training_set_neg)
    
    num_correct = 0
    num_total = 0
    
    # test on positive ground truth
    for document in testing_set_pos:
        prediction = classify_text(document, nb_model)
        if prediction == 'POSITIVE':
            num_correct += 1
        num_total += 1
        
    # test on negative ground truth
    for document in testing_set_neg:
        prediction = classify_text(document, nb_model)
        if prediction == 'NEGATIVE':
            num_correct += 1
        num_total += 1
        
    acc = (num_correct / num_total) * 100
    print(f"Accuracy: {acc:.2f}%")
    
    # Interactive Console
    print("\nType a sentence to test (or type 'exit' to quit):")
    while True:
        try:
            user_input = input("> ")
            if user_input.strip().lower() == 'exit':
                break
            result = classify_text(user_input, nb_model)
            print(f"Prediction: {result}")
        except (EOFError, KeyboardInterrupt):
            break
            
if __name__ == '__main__':
    run_classifier()
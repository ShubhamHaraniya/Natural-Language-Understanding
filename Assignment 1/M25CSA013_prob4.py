"""
NLP Assignment 1 - Problem 4
Student ID: M25CSA013
Sports vs Politics Classifier
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from wordcloud import WordCloud
import os

# Create directories for output if they don't exist
if not os.path.exists('docs/images'):
    os.makedirs('docs/images')

def load_data():
    # loading the 20 newsgroups dataset
    # we just need sports and politics categories
    categories = [
        'rec.sport.baseball', 'rec.sport.hockey',
        'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc'
    ]
    
    print("Loading 20 Newsgroups dataset...")
    # fetching train and test sets separately
    train_data = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
    test_data = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)
    
    # helper to convert labels to binary
    # 0 for Sports, 1 for Politics
    
    def get_binary_label(target_idx, target_names):
        name = target_names[target_idx]
        if name.startswith('rec.sport'):
            return 'Sports'
        else:
            return 'Politics'

    y_train_bin = [get_binary_label(y, train_data.target_names) for y in train_data.target]
    y_test_bin = [get_binary_label(y, test_data.target_names) for y in test_data.target]
    
    return train_data.data, y_train_bin, test_data.data, y_test_bin

def analyze_data(X_train, y_train):
    # simple data visualization
    print("Performing data analysis...")
    df = pd.DataFrame({'text': X_train, 'label': y_train})
    
    # 1. checking class balance
    plt.figure(figsize=(8, 6))
    sns.countplot(x='label', data=df)
    plt.title('Class Distribution (Training Set)')
    plt.savefig('docs/images/class_distribution.png')
    plt.close()
    
    # 2. generating word clouds
    for label in ['Sports', 'Politics']:
        text = " ".join(df[df['label'] == label]['text'])
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Word Cloud - {label}')
        plt.savefig(f'docs/images/wordcloud_{label.lower()}.png')
        plt.close()
        
    # 3. checking most frequent words
    cv = CountVectorizer(stop_words='english', max_features=20)
    
    for label in ['Sports', 'Politics']:
        subset = df[df['label'] == label]['text']
        bow = cv.fit_transform(subset)
        word_freq = dict(zip(cv.get_feature_names_out(), bow.toarray().sum(axis=0)))
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        words, counts = zip(*sorted_words)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(counts), y=list(words))
        plt.title(f'Top 20 Words - {label}')
        plt.xlabel('Frequency')
        plt.savefig(f'docs/images/top_words_{label.lower()}.png')
        plt.close()

def train_evaluate(X_train, y_train, X_test, y_test):
    # training different models and comparing them
    experiments = [
        ("BoW + NB", CountVectorizer(stop_words='english'), MultinomialNB()),
        ("BoW + SVM", CountVectorizer(stop_words='english'), LinearSVC(random_state=42, dual='auto')),
        ("BoW + LR", CountVectorizer(stop_words='english'), LogisticRegression(max_iter=1000, random_state=42)),
        
        ("TF-IDF + NB", TfidfVectorizer(stop_words='english'), MultinomialNB()),
        ("TF-IDF + SVM", TfidfVectorizer(stop_words='english'), LinearSVC(random_state=42, dual='auto')),
        ("TF-IDF + LR", TfidfVectorizer(stop_words='english'), LogisticRegression(max_iter=1000, random_state=42)),
        
        ("N-gram(1,2) + NB", TfidfVectorizer(ngram_range=(1, 2), stop_words='english'), MultinomialNB()),
        ("N-gram(1,2) + SVM", TfidfVectorizer(ngram_range=(1, 2), stop_words='english'), LinearSVC(random_state=42, dual='auto')),
        ("N-gram(1,2) + LR", TfidfVectorizer(ngram_range=(1, 2), stop_words='english'), LogisticRegression(max_iter=1000, random_state=42))
    ]
    
    results = []
    best_acc = 0
    best_model_name = ""
    best_y_pred = None
    
    output_file = "model_results.txt"
    with open(output_file, "w") as f:
        f.write("Model Evaluation Results\n")
        f.write("========================\n\n")

    print(f"Training {len(experiments)} models...")
    
    for name, vectorizer, model in experiments:
        print(f"Running experiment: {name}")
        
        # training the model
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)
        
        acc = accuracy_score(y_test, y_pred)
        results.append({'Model': name, 'Accuracy': acc})
        
        report = classification_report(y_test, y_pred)
        
        with open(output_file, "a") as f:
            f.write(f"--- {name} ---\n")
            f.write(f"Accuracy: {acc:.4f}\n")
            f.write(report + "\n\n")
            
        if acc > best_acc:
            best_acc = acc
            best_model_name = name
            best_y_pred = y_pred

    # plotting results
    res_df = pd.DataFrame(results)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Accuracy', y='Model', data=res_df)
    plt.title('Model Comparison - Accuracy')
    plt.tight_layout()
    plt.savefig('docs/images/model_comparison.png')
    plt.close()
    
    # showing confusion matrix for the best model
    cm = confusion_matrix(y_test, best_y_pred, labels=['Sports', 'Politics'])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Sports', 'Politics'], yticklabels=['Sports', 'Politics'])
    plt.title(f'Confusion Matrix ({best_model_name})')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig('docs/images/confusion_matrix_best.png')
    plt.close()
    
    print(f"Training complete. Results saved to {output_file} and images to docs/images/")

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_data()
    analyze_data(X_train, y_train)
    train_evaluate(X_train, y_train, X_test, y_test)

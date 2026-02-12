"""
NLP Assignment 1 - Problem 4
Student ID: M25CSA013
Sports vs Politics Classifier
"""

# download opendatasets if not already installed
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os
import json
import opendatasets as od

# Create directories for output if they don't exist
if not os.path.exists('docs/images'):
    os.makedirs('docs/images')

def download_dataset():
    """Download News Category Dataset from Kaggle if not already present"""
    dataset_path = 'news-category-dataset/News_Category_Dataset_v3.json'
    
    if not os.path.exists(dataset_path):
        print("Downloading News Category Dataset from Kaggle...")
        print("Note: You may need to provide your Kaggle credentials.")
        print("You can find them at: https://www.kaggle.com/settings/account")
        
        try:
            # Download the dataset
            od.download('https://www.kaggle.com/datasets/rmisra/news-category-dataset')
            print("Dataset downloaded successfully!")
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            print("Please manually download News_Category_Dataset_v3.json from:")
            print("https://www.kaggle.com/datasets/rmisra/news-category-dataset")
            raise
    else:
        print("Dataset already exists locally.")
    
    return dataset_path

def load_data():
    # loading the News Category Dataset from Kaggle
    # Dataset: https://www.kaggle.com/datasets/rmisra/news-category-dataset
    # Using only POLITICS and SPORTS categories
    # Balancing by sampling same number from Politics as Sports
    
    print("Loading News Category Dataset...")
    
    # Download dataset if needed
    dataset_path = download_dataset()
    
    # Load the JSON dataset
    data = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    
    df = pd.DataFrame(data)
    print(f"Total records in dataset: {len(df)}")
    print(f"Categories found: {df['category'].unique()[:10]}...")  # Show first 10
    
    # Filter for only POLITICS and SPORTS categories
    df_politics = df[df['category'] == 'POLITICS'].copy()
    df_sports = df[df['category'] == 'SPORTS'].copy()
    
    print(f"\nOriginal class distribution:")
    print(f"Politics samples: {len(df_politics)}")
    print(f"Sports samples: {len(df_sports)}")
    
    # Balance the dataset by sampling from Politics to match Sports count
    n_sports = len(df_sports)
    df_politics_sampled = df_politics.sample(n=n_sports, random_state=42)
    
    # Combine balanced datasets
    df_balanced = pd.concat([df_politics_sampled, df_sports], ignore_index=True)
    
    # Create text field by combining headline and short_description
    df_balanced['text'] = df_balanced['headline'] + ' ' + df_balanced['short_description']
    
    # Map category to simpler labels
    df_balanced['label'] = df_balanced['category'].map({'POLITICS': 'Politics', 'SPORTS': 'Sports'})
    
    print(f"\nBalanced dataset:")
    print(f"Total samples: {len(df_balanced)}")
    print(f"Politics samples: {sum(df_balanced['label'] == 'Politics')}")
    print(f"Sports samples: {sum(df_balanced['label'] == 'Sports')}")
    
    # Split into train and test sets for evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        df_balanced['text'].tolist(),
        df_balanced['label'].tolist(),
        test_size=0.2,
        random_state=42,
        stratify=df_balanced['label'].tolist()
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    return X_train, y_train, X_test, y_test

def analyze_data(X_train, y_train):
    # simple data visualization
    print("Performing data analysis...")
    df = pd.DataFrame({'text': X_train, 'label': y_train})
    
    # Set style for better looking plots
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 11
    
    # 1. Class distribution with enhanced styling
    plt.figure(figsize=(10, 6))
    colors = ['#FF6B6B', '#4ECDC4']
    ax = sns.countplot(x='label', data=df, palette=colors, edgecolor='black', linewidth=1.5)
    plt.title('Class Distribution (Training Set)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Category', fontsize=13, fontweight='bold')
    plt.ylabel('Number of Samples', fontsize=13, fontweight='bold')
    
    # Add count labels on top of bars
    for container in ax.containers:
        ax.bar_label(container, fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('docs/images/class_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Top frequent words with gradient colors
    cv = CountVectorizer(stop_words='english', max_features=20)
    
    for label in ['Sports', 'Politics']:
        subset = df[df['label'] == label]['text']
        bow = cv.fit_transform(subset)
        word_freq = dict(zip(cv.get_feature_names_out(), bow.toarray().sum(axis=0)))
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        words, counts = zip(*sorted_words)
        
        # Create gradient color palette
        color_palette = sns.color_palette("rocket" if label == 'Politics' else "mako", n_colors=20)
        
        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(words)), counts, color=color_palette, edgecolor='black', linewidth=0.8)
        plt.yticks(range(len(words)), words, fontsize=11)
        plt.xlabel('Frequency', fontsize=13, fontweight='bold')
        plt.ylabel('Words', fontsize=13, fontweight='bold')
        plt.title(f'Top 20 Most Frequent Words - {label}', fontsize=16, fontweight='bold', pad=20)
        plt.gca().invert_yaxis()
        
        # Add value labels on bars
        for i, (bar, count) in enumerate(zip(bars, counts)):
            plt.text(count + max(counts)*0.01, i, f'{int(count)}', 
                    va='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'docs/images/top_words_{label.lower()}.png', dpi=300, bbox_inches='tight')
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

    # plotting results with enhanced styling
    res_df = pd.DataFrame(results)
    
    plt.figure(figsize=(12, 8))
    colors = sns.color_palette('viridis', n_colors=len(res_df))
    bars = plt.barh(res_df['Model'], res_df['Accuracy'], color=colors, edgecolor='black', linewidth=1.2)
    
    plt.xlabel('Accuracy', fontsize=13, fontweight='bold')
    plt.ylabel('Model', fontsize=13, fontweight='bold')
    plt.title('Model Performance Comparison', fontsize=16, fontweight='bold', pad=20)
    plt.xlim(0, 1.0)
    
    # Add accuracy values on bars
    for i, (bar, acc) in enumerate(zip(bars, res_df['Accuracy'])):
        plt.text(acc + 0.01, i, f'{acc:.4f}', va='center', fontsize=10, fontweight='bold')
    
    # Add grid for better readability
    plt.grid(axis='x', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('docs/images/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Confusion matrix with enhanced styling
    cm = confusion_matrix(y_test, best_y_pred, labels=['Sports', 'Politics'])
    
    plt.figure(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', 
                xticklabels=['Sports', 'Politics'], 
                yticklabels=['Sports', 'Politics'],
                cbar_kws={'label': 'Count'},
                linewidths=2, linecolor='black',
                annot_kws={'fontsize': 14, 'fontweight': 'bold'})
    
    plt.title(f'Confusion Matrix - {best_model_name}', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Actual Label', fontsize=13, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('docs/images/confusion_matrix_best.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training complete. Results saved to {output_file} and images to docs/images/")

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_data()
    analyze_data(X_train, y_train)
    train_evaluate(X_train, y_train, X_test, y_test)

# Text Classification Report: Sports vs Politics

**Author**: Shubham Haraniya (M25CSA013)  
**Date**: February 1, 2026

## 1. Introduction
This project aims to build a machine learning classifier to distinguish between "Sports" and "Politics" documents. Text classification is a fundamental task in Natural Language Processing (NLP) with applications in news categorization, sentiment analysis, and spam detection. We compare three supervised learning algorithms—Naive Bayes, Support Vector Machines (SVM), and Logistic Regression—using Bag of Words (BoW), TF-IDF, and N-gram feature representations.

## 2. Data Collection
We utilized the **20 Newsgroups** dataset, a standard benchmark for text classification. Specifically, we selected the following categories to form our binary classification task:

*   **Sports**: `rec.sport.baseball`, `rec.sport.hockey`
*   **Politics**: `talk.politics.guns`, `talk.politics.mideast`, `talk.politics.misc`

The dataset handling was facilitated by `sklearn.datasets.fetch_20newsgroups`.

## 3. Dataset Description and Analysis
### 3.1 Class Distribution
The dataset is split into training and testing sets. We analyzed the balance between the two classes (Sports and Politics).
(See `docs/images/class_distribution.png` for visualization)

### 3.2 Top Frequent Words
We extracted the top 20 most frequent words for each category to understand the vocabulary.
*   **Sports**: Frequent words include ... [TO BE FILLED FROM RESULTS]
*   **Politics**: Frequent words include ... [TO BE FILLED FROM RESULTS]

(See `docs/images/top_words_sports.png` and `docs/images/top_words_politics.png`)

### 3.3 Word Clouds
Word clouds were generated to visualize the most prominent terms.
(See `docs/images/wordcloud_sports.png` and `docs/images/wordcloud_politics.png`)

## 4. Methodology
### 4.1 Preprocessing
*   **Tokenization**: Splitting text into words.
*   **Stopword Removal**: Removing common English words (e.g., "the", "and") using Scikit-Learn's built-in list.
*   **Case Folding**: Converting text to lowercase.

### 4.2 Feature Representation
*   **Bag of Words (BoW)**: Counts word occurrences.
*   **TF-IDF**: Term Frequency-Inverse Document Frequency, which weighs words by their importance.
*   **N-grams**: We explored Bigrams (combinations of 2 adjacent words) to capture context (e.g., "white house").

### 4.3 Machine Learning Models
1.  **Multinomial Naive Bayes (MNB)**: A probabilistic classifier based on Bayes' theorem, well-suited for discrete text counts.
2.  **Linear Support Vector Machine (SVC)**: Finds the optimal hyperplane to separate classes; effective in high-dimensional spaces.
3.  **Logistic Regression (LR)**: Models probability of class membership using a logistic function.

## 5. Experiments and Results

We conducted multiple experiments comparing combinations of features and models.

| Experiment | Feature | Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | BoW | Naive Bayes | [VAL] | [VAL] | [VAL] | [VAL] |
| 2 | BoW | SVM | [VAL] | [VAL] | [VAL] | [VAL] |
| 3 | BoW | LR | [VAL] | [VAL] | [VAL] | [VAL] |
| 4 | TF-IDF | Naive Bayes | [VAL] | [VAL] | [VAL] | [VAL] |
| 5 | TF-IDF | SVM | [VAL] | [VAL] | [VAL] | [VAL] |
| 6 | TF-IDF | LR | [VAL] | [VAL] | [VAL] | [VAL] |
| 7 | TF-IDF (1,2-gram) | SVM | [VAL] | [VAL] | [VAL] | [VAL] |

### 5.1 Quantitative Comparisons
The comparative performance is shown in the graph below:
(See `docs/images/model_comparison.png`)

**Confusion Matrix**: Evalaulating our best performing model:
(See `docs/images/confusion_matrix_best.png`)

## 6. Discussion
[TO BE FILLED BASED ON RESULTS - e.g. SVM with TF-IDF likely performed best due to...]

## 7. Limitations
*   **Dataset Size**: The 20 Newsgroups subset is relatively small (~5000 docs).
*   **Static Vocabulary**: The models cannot handle Out-Of-Vocabulary (OOV) words not seen during training.
*   **Scalability**: SVMs can be slow to train on very large corpora.

## 8. Conclusion
We successfully built a classifier distinguishing Sports from Politics. [Final Sentence on best model].

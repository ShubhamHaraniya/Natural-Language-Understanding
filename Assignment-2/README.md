# NLP Assignment 2

This repository contains the source code, reports, and models for NLP Assignment 2. The assignment is divided into two problems: Problem 1 (Word Embeddings) and Problem 2 (Character-Level RNN Name Generation).

## Project Structure

- `Problem 1/` - Word2Vec (Skip-gram and CBOW) implementations built purely in NumPy from scratch to train word embeddings on IIT Jodhpur's academic documents. Contains preprocessing, training, evaluation (nearest neighbors, analogies), and visualisation scripts.
- `Problem 2/` - Character-level RNNs (Vanilla RNN, Bidirectional LSTM, and RNN+Attention) built in PyTorch from scratch to generate Indian names. Includes training, sampling, and qualitative/quantitative evaluations.

## Running the Code

See the respective `README.md` files in `Problem 1/` and `Problem 2/` for detailed structure and instructions on how to reproduce the training and evaluation steps.

### High-level workflow for Problem 1
```bash
cd "Problem 1"
python preprocess.py
python word2vec.py
python analyze.py
python visualize.py
```

### High-level workflow for Problem 2
```bash
cd "Problem 2"
python train.py
python generate.py
python evaluate.py
```

## Report and Corpus

The generated final `report.pdf` and the preprocessed `corpus.txt` required for the assignment submission are provided in the respective submission zip file format.


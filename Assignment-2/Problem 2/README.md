# Character-Level Name Generation Using RNN Variants

A deep learning assignment that builds three different recurrent neural network architectures from scratch and uses them to generate Indian names, one character at a time.

---

## What this project does

The idea is pretty simple — you give the model thousands of Indian names to learn from, and it figures out the character-level patterns that make a name sound Indian. Then it generates new names on its own, predicting one character at a time starting from nothing.

We compare three architectures to see how the model design affects the quality of what gets generated.

---

## Project Structure

```
Problem 2/
│
├── TrainingNames.txt          # 1000 Indian names used for training
│
├── dataset.py                 # Vocabulary builder + data loading utilities
│
├── models/
│   ├── vanilla_rnn.py         # Model 1 — simplest RNN, built from scratch
│   ├── blstm.py               # Model 2 — bidirectional LSTM, manual gates
│   └── rnn_attention.py       # Model 3 — RNN with self-attention
│
├── train.py                   # Trains all three models and saves weights
├── generate.py                # Generates 100 names per model
├── evaluate.py                # Computes metrics and produces plots
│
├── saved_models/              # Saved weights after training
├── generated_samples/         # 100 generated names per model (txt files)
│   ├── vanilla_rnn_names.txt
│   ├── blstm_names.txt
│   └── rnn_attention_names.txt
│
├── evaluation_results/        # Plots + novel name lists
│   ├── novelty_diversity_comparison.png
│   ├── name_length_distribution.png
│   ├── novel_vs_copied.png
│   ├── char_frequency.png
│   ├── vanilla_rnn_novel_names.txt
│   ├── blstm_novel_names.txt
│   └── rnn_attention_novel_names.txt
│
├── training_loss.png          # Loss curves for all three models
```

---

## How to run it

> Make sure you have PyTorch and matplotlib installed.

```bash
# Step 1: train all three models (uses GPU if available)
python train.py

# Step 2: generate 100 names from each model
python generate.py

# Step 3: evaluate and produce all plots
python evaluate.py
```

That's it. Each step builds on the previous one.

---

## The three models

All models are built **from scratch** — no `nn.RNN` or `nn.LSTM` from PyTorch. The weight matrices and recurrence equations are coded manually.

### Vanilla RNN
The simplest possible recurrent model. At each character step, it updates a hidden state using:

```
h_t = tanh(W_ih · x_t + W_hh · h_{t-1} + b)
```

**28,980 parameters.** Works surprisingly well for short names but struggles with longer ones due to vanishing gradients.

### Bidirectional LSTM (BLSTM)
Same idea as the RNN but with LSTM gating (forget, input, output gates) and reads the sequence in both directions during training. Only the forward direction is used at generation time.

**186,600 parameters.** Produces the most natural-sounding names. The gating mechanism is great at remembering which characters were used earlier in the name.

### RNN with Bahdanau Self-Attention
At each timestep, the model doesn't just look at its current hidden state — it also looks back at all its previous hidden states and decides which ones matter most right now. This is the attention part.

**68,532 parameters.** Adds context-awareness but doesn't dramatically outperform BLSTM for such short sequences.

---

## Results

| Model | Novelty | Diversity | Params |
|---|---|---|---|
| Vanilla RNN | 14% | 98% | 28,980 |
| BLSTM | 7% | 95% | 186,600 |
| RNN + Attention | 7% | 96% | 68,532 |

**The interesting finding:** Vanilla RNN has the highest novelty (14%), but when you look at those novel names closely — *Jagnkoh, Gajasviram, Udna* — they're novel because they're broken, not because they're creative. BLSTM's novel names — *Raya, Aarchan, Sanjoya* — are actually pronounceable and feel like real names.

> **Higher novelty % ≠ better generation quality.**

---

## Evaluation plots

Running `evaluate.py` generates four plots in `evaluation_results/`:

- **novelty_diversity_comparison.png** — bar chart comparing both metrics across models
- **name_length_distribution.png** — histogram showing how long the generated names tend to be
- **novel_vs_copied.png** — pie charts showing what fraction of names were original vs copied
- **char_frequency.png** — which characters each model uses most

It also saves the novel names (ones not in the training set) for each model into separate text files, so you can inspect them directly.

---

## Hyperparameters (same for all models)

| Setting | Value |
|---|---|
| Embedding dim | 32 |
| Hidden size | 128 |
| Learning rate | 0.003 |
| Batch size | 64 |
| Epochs | 100 |
| Optimizer | Adam |
| Temperature (generation) | 0.8 |

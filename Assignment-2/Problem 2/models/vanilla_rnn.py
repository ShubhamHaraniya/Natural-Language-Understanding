import torch
import torch.nn as nn
import math


class VanillaRNN(nn.Module):
    """
    Basic RNN cell written by hand — no nn.RNN.
    h_t = tanh(W_ih * x_t + W_hh * h_{t-1} + b)
    """

    def __init__(self, vocab_size, embed_dim=32, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.embed = nn.Embedding(vocab_size, embed_dim)

        # RNN weights
        self.W_ih = nn.Parameter(torch.randn(embed_dim, hidden_dim) / math.sqrt(embed_dim))
        self.W_hh = nn.Parameter(torch.randn(hidden_dim, hidden_dim) / math.sqrt(hidden_dim))
        self.b_h  = nn.Parameter(torch.zeros(hidden_dim))

        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        batch, seq_len = x.shape
        h = torch.zeros(batch, self.hidden_dim, device=x.device)
        embeds = self.embed(x)

        outputs = []
        for t in range(seq_len):
            h = torch.tanh(embeds[:, t] @ self.W_ih + h @ self.W_hh + self.b_h)
            outputs.append(self.fc(h))

        return torch.stack(outputs, dim=1)

    def generate(self, vocab, max_len=20, temperature=0.8):
        """sample one name character by character starting from SOS"""
        self.eval()
        device = next(self.parameters()).device
        h = torch.zeros(1, self.hidden_dim, device=device)
        idx = vocab.sos_idx
        name_chars = []

        with torch.no_grad():
            for _ in range(max_len):
                x = self.embed(torch.tensor([[idx]], device=device)).squeeze(1)
                h = torch.tanh(x @ self.W_ih + h @ self.W_hh + self.b_h)
                # temperature > 1 → more random, < 1 → more conservative
                logits = self.fc(h) / temperature
                probs = torch.softmax(logits, dim=-1)
                idx = torch.multinomial(probs.squeeze(0), 1).item()
                if idx == vocab.eos_idx:
                    break
                name_chars.append(vocab.idx2ch[idx])

        self.train()
        return ''.join(name_chars)

    def description(self):
        return (
            "Vanilla RNN\n"
            "Architecture: Embedding -> Manual RNN Cell (tanh) -> Linear\n"
            f"Hidden size: {self.hidden_dim}, Embedding: {self.embed.embedding_dim}\n"
            "Recurrence: h_t = tanh(W_ih * x_t + W_hh * h_{t-1} + b)"
        )

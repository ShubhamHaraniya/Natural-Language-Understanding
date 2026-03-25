import torch
import torch.nn as nn
import math


class RNNAttention(nn.Module):
    """
    RNN with Bahdanau self-attention.
    At each step, instead of relying only on the current hidden state,
    the model looks back at all previous hidden states and picks the most relevant ones.
    Steps:
      1. compute h_t from input + previous hidden
      2. score each past h_j against h_t
      3. take a weighted average of past states → context c_t
      4. predict next char from [h_t, c_t]
    """

    def __init__(self, vocab_size, embed_dim=32, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embed = nn.Embedding(vocab_size, embed_dim)

        # RNN weights
        self.W_ih = nn.Parameter(torch.randn(embed_dim, hidden_dim) / math.sqrt(embed_dim))
        self.W_hh = nn.Parameter(torch.randn(hidden_dim, hidden_dim) / math.sqrt(hidden_dim))
        self.b_h  = nn.Parameter(torch.zeros(hidden_dim))

        # Bahdanau attention weights
        self.W_attn = nn.Linear(hidden_dim, hidden_dim, bias=False)  # scores past states
        self.U_attn = nn.Linear(hidden_dim, hidden_dim, bias=False)  # scores current state
        self.v_attn = nn.Linear(hidden_dim, 1, bias=False)           # produces scalar score

        self.fc = nn.Linear(hidden_dim * 2, vocab_size)

    def _attend(self, current_h, past_states):
        """weighted sum of past hidden states using current h as the query"""
        cur_proj  = self.U_attn(current_h).unsqueeze(1)
        past_proj = self.W_attn(past_states)
        energy    = self.v_attn(torch.tanh(past_proj + cur_proj))
        weights   = torch.softmax(energy.squeeze(-1), dim=-1)
        context   = (weights.unsqueeze(-1) * past_states).sum(dim=1)
        return context

    def forward(self, x):
        batch, seq_len = x.shape
        embeds = self.embed(x)
        h = torch.zeros(batch, self.hidden_dim, device=x.device)

        all_hidden = []
        outputs = []

        for t in range(seq_len):
            h = torch.tanh(embeds[:, t] @ self.W_ih + h @ self.W_hh + self.b_h)
            all_hidden.append(h)

            if t == 0:
                # no history yet — just use h as-is
                context = h
            else:
                past = torch.stack(all_hidden[:t], dim=1)
                context = self._attend(h, past)

            combined = torch.cat([h, context], dim=-1)
            outputs.append(self.fc(combined))

        return torch.stack(outputs, dim=1)

    def generate(self, vocab, max_len=20, temperature=0.8):
        """same as training — model attends over its own history while generating"""
        self.eval()
        device = next(self.parameters()).device
        h = torch.zeros(1, self.hidden_dim, device=device)
        idx = vocab.sos_idx
        name_chars = []
        all_hidden = []

        with torch.no_grad():
            for step in range(max_len):
                x = self.embed(torch.tensor([[idx]], device=device)).squeeze(1)
                h = torch.tanh(x @ self.W_ih + h @ self.W_hh + self.b_h)
                all_hidden.append(h)

                if step == 0:
                    context = h
                else:
                    past = torch.stack(all_hidden[:step], dim=1)
                    context = self._attend(h, past)

                combined = torch.cat([h, context], dim=-1)
                logits = self.fc(combined) / temperature
                probs  = torch.softmax(logits, dim=-1)
                idx    = torch.multinomial(probs.squeeze(0), 1).item()

                if idx == vocab.eos_idx:
                    break
                name_chars.append(vocab.idx2ch[idx])

        self.train()
        return ''.join(name_chars)

    def description(self):
        return (
            "RNN with Bahdanau Self-Attention\n"
            "Architecture: RNN Cell -> Self-Attention over past hidden states -> Linear\n"
            f"Hidden size: {self.hidden_dim}, Embedding: {self.embed.embedding_dim}\n"
            "Attention: additive (Bahdanau) attending over decoder's own history"
        )

import torch
import torch.nn as nn
import math


class LSTMCell(nn.Module):
    """
    Manual LSTM cell — forget, input, and output gates packed into one matrix.
    f_gate = what to forget from cell state
    i_gate = what new info to write in
    o_gate = what part of the cell to expose as hidden state
    """

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        # all four gates in one matrix for efficiency
        self.W = nn.Parameter(torch.randn(input_dim + hidden_dim, 4 * hidden_dim) / math.sqrt(input_dim + hidden_dim))
        self.b = nn.Parameter(torch.zeros(4 * hidden_dim))

    def forward(self, x, h_prev, c_prev):
        combined = torch.cat([x, h_prev], dim=-1)
        gates = combined @ self.W + self.b

        f_gate = torch.sigmoid(gates[:, :self.hidden_dim])
        i_gate = torch.sigmoid(gates[:, self.hidden_dim:2*self.hidden_dim])
        c_cand = torch.tanh(gates[:, 2*self.hidden_dim:3*self.hidden_dim])
        o_gate = torch.sigmoid(gates[:, 3*self.hidden_dim:])

        c = f_gate * c_prev + i_gate * c_cand
        h = o_gate * torch.tanh(c)
        return h, c


class BLSTM(nn.Module):
    """
    Bidirectional LSTM — reads both left-to-right and right-to-left during training.
    Only the forward direction is used at generation time (can't peek ahead).
    Training loss = forward loss + 0.5 * bidirectional loss.
    """

    def __init__(self, vocab_size, embed_dim=32, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embed = nn.Embedding(vocab_size, embed_dim)

        self.fwd_cell = LSTMCell(embed_dim, hidden_dim)
        self.bwd_cell = LSTMCell(embed_dim, hidden_dim)

        self.fc    = nn.Linear(hidden_dim, vocab_size)       # used at generation
        self.fc_bi = nn.Linear(hidden_dim * 2, vocab_size)   # used during training

    def forward(self, x):
        batch, seq_len = x.shape
        embeds = self.embed(x)
        device = x.device

        # left→right
        h_f = torch.zeros(batch, self.hidden_dim, device=device)
        c_f = torch.zeros(batch, self.hidden_dim, device=device)
        fwd_out = []
        for t in range(seq_len):
            h_f, c_f = self.fwd_cell(embeds[:, t], h_f, c_f)
            fwd_out.append(h_f)

        # right→left
        h_b = torch.zeros(batch, self.hidden_dim, device=device)
        c_b = torch.zeros(batch, self.hidden_dim, device=device)
        bwd_out = [None] * seq_len
        for t in range(seq_len - 1, -1, -1):
            h_b, c_b = self.bwd_cell(embeds[:, t], h_b, c_b)
            bwd_out[t] = h_b

        fwd_outputs = []
        bi_outputs  = []
        for t in range(seq_len):
            fwd_outputs.append(self.fc(fwd_out[t]))
            bi_outputs.append(self.fc_bi(torch.cat([fwd_out[t], bwd_out[t]], dim=-1)))

        return torch.stack(fwd_outputs, dim=1), torch.stack(bi_outputs, dim=1)

    def generate(self, vocab, max_len=20, temperature=0.8):
        """forward-only at inference"""
        self.eval()
        device = next(self.parameters()).device
        h = torch.zeros(1, self.hidden_dim, device=device)
        c = torch.zeros(1, self.hidden_dim, device=device)
        idx = vocab.sos_idx
        name_chars = []

        with torch.no_grad():
            for _ in range(max_len):
                x = self.embed(torch.tensor([[idx]], device=device)).squeeze(1)
                h, c = self.fwd_cell(x, h, c)
                logits = self.fc(h) / temperature
                probs  = torch.softmax(logits, dim=-1)
                idx    = torch.multinomial(probs.squeeze(0), 1).item()
                if idx == vocab.eos_idx:
                    break
                name_chars.append(vocab.idx2ch[idx])

        self.train()
        return ''.join(name_chars)

    def description(self):
        return (
            "Bidirectional LSTM\n"
            "Architecture: Embedding -> Forward LSTM Cell + Backward LSTM Cell -> Concat -> Linear\n"
            f"Hidden size: {self.hidden_dim} per direction, Embedding: {self.embed.embedding_dim}\n"
            "Gates: forget, input, output\n"
            "Training: combined loss from bidirectional + forward-only paths\n"
            "Generation: forward direction only"
        )

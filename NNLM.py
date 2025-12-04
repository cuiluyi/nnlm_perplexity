import torch
import torch.nn as nn


# feedforward NNLM
class FFNModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_size: int,
        hidden_size: int,
        context_size: int,
    ):
        super().__init__()
        # Embedding layer: convert word indices to dense vectors
        self.embeddings = nn.Embedding(vocab_size, embed_size)

        # First linear layer: maps concatenated embeddings to hidden layer
        self.linear1 = nn.Linear(embed_size * context_size, hidden_size)

        # Second linear layer: maps hidden layer to vocabulary logits
        self.linear2 = nn.Linear(hidden_size, vocab_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        inputs: (batch_size, context_size)
        outputs: (batch_size, vocab_size)
        """
        # Look up embeddings for each word in the context
        embeds = self.embeddings(inputs)  # (batch_size, context_size, embed_size)

        # Flatten (concatenate embeddings of all context words)
        x = embeds.view(embeds.size(0), -1)  # (batch_size, context_size * embed_size)

        # Nonlinear hidden layer
        h = torch.tanh(self.linear1(x))  # (batch_size, hidden_size)

        # Compute logits
        outputs = self.linear2(h)  # (batch_size, vocab_size)

        # CrossEntropyLoss in PyTorch (applies Softmax)
        # nn.LogSoftmax + nn.NLLLoss (negative log likelihood loss)
        # -> No Softmax in last layer!
        return outputs


# Recurrent NNLM
class RNNModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_size: int,
        hidden_size: int,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(
            embed_size,
            hidden_size,
            batch_first=True,
        )
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        inputs: (batch_size, seq_len)
        outputs: (batch_size, vocab_size)
        """
        embeds = self.embedding(inputs)  # (batch_size, seq_len, embed_size)

        h, _ = self.rnn(embeds)  # (batch_size, seq_len, hidden_size)

        outputs = self.linear(h)  # (batch_size, seq_len, vocab_size)

        # CrossEntropyLoss in PyTorch (applies Softmax)
        # nn.LogSoftmax + nn.NLLLoss (negative log likelihood loss)
        # -> No Softmax in last layer!
        # Use the last time step's output for classification
        return outputs[:, -1, :]


# Self-Attention NNLM
class SelfAttentionNNLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_size: int,
        dim_feedforward: int = 2048,
        num_heads: int = 1,
    ):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_size)
        self.attention = nn.MultiheadAttention(
            embed_size,
            num_heads,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(embed_size)
        # the default value of dim_feedforward in PyTorch Transformer is 2048
        # you can alse set it to other values, like 2 * embed_size
        self.ffn = nn.Sequential(
            nn.Linear(embed_size, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, embed_size),
        )
        self.norm2 = nn.LayerNorm(embed_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        inputs: (batch_size, seq_len)
        outputs: (batch_size, vocab_size)
        """
        embeds = self.embeddings(inputs)  # (batch_size, seq_len, embed_size)

        # Self-attention mechanism
        attn_output, _ = self.attention(embeds, embeds, embeds)  # (batch_size, seq_len, embed_size)

        # Add & Norm
        attn_output = self.norm1(attn_output + embeds)  # (batch_size, seq_len, embed_size)

        # Feedforward layer
        x = self.ffn(attn_output)  # (batch_size, seq_len, embed_size)

        # Add & Norm
        outputs = self.norm2(x + attn_output)  # (batch_size, seq_len, embed_size)

        # CrossEntropyLoss in PyTorch (applies Softmax)
        # nn.LogSoftmax + nn.NLLLoss (negative log likelihood loss)
        # -> No Softmax in last layer!
        # Use the last time step's output for classification
        return outputs[:, -1, :]
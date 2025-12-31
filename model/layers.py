import math
from typing import Tuple

import torch
from jaxtyping import Float
from rotary_embedding_torch import RotaryEmbedding
from torch import nn, Tensor
from torch.nn import ModuleDict, LayerNorm
from torch.nn.functional import softmax, scaled_dot_product_attention

from model.config import TransformerConfig

FLOAT_MIN = float('-inf')


class RotaryPositionalEmbedding(nn.Module):
    rotation_matrix: Float[Tensor, "EMBEDDING_DIM EMBEDDING_DIM"]
    positional_embedding: Float[Tensor, "SEQUENCE_LENGTH EMBEDDING_DIM"]

    def __init__(self, config: TransformerConfig):
        super().__init__()

        self.rotation_matrix = torch.zeros(config.embedding_dim, config.embedding_dim)
        for i in range(config.embedding_dim):
            for j in range(config.embedding_dim):
                self.rotation_matrix[i, j] = torch.cos(torch.scalar_tensor(i * j * 0.01))

        self.positional_embedding = torch.zeros(config.sequence_length, config.embedding_dim)
        for i in range(config.sequence_length):
            for j in range(config.embedding_dim):
                self.positional_embedding[i, j] = torch.cos(torch.scalar_tensor(i * j * 0.01))

    def forward(self, x: Float[Tensor, "BATCH_SIZE SEQUENCE_LENGTH EMBEDDING_DIM"]):
        x += self.positional_embedding

        x = x @ self.rotation_matrix

        return x


class PositionalEmbedding(nn.Module):
    positional_embedding: Float[Tensor, "SEQUENCE_LENGTH EMBEDDING_DIM"]

    def __init__(self, config: TransformerConfig):
        super().__init__()
        pos = torch.arange(0, config.sequence_length, dtype=torch.float32).unsqueeze(1)

        i = torch.arange(0, config.embedding_dim, 2, dtype=torch.float32) / config.embedding_dim

        angle_freq = torch.exp(i * (-torch.log(torch.tensor(10000.0))))

        pos_encoding_sin = torch.sin(pos * angle_freq)
        pos_encoding_cos = torch.cos(pos * angle_freq)

        positional_embedding = torch.cat([pos_encoding_sin, pos_encoding_cos], dim=-1)
        self.positional_embedding = nn.Parameter(data=positional_embedding, requires_grad=False)

    def forward(self, x: Float[Tensor, "BATCH_SIZE SEQUENCE_LENGTH EMBEDDING_DIM"]):
        seq_len = x.size(1)

        x = x + self.positional_embedding[:seq_len]

        return x


class FFN(nn.Module):

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.fully_connected = nn.Linear(config.embedding_dim, 4 * config.embedding_dim)
        self.gelu = nn.GELU()
        self.output_projection = nn.Linear(4 * config.embedding_dim, config.embedding_dim)

    def forward(self, x):
        x = self.fully_connected(x)
        x = self.gelu(x)
        x = self.output_projection(x)
        return x


class MultiHeadedAttention(nn.Module):
    in_project: nn.Linear
    out_project: nn.Linear
    num_attention_heads: int
    embedding_dim: int
    attention_dim: int
    causal_mask: Tensor
    flash_attention: bool
    rotary_embedding: RotaryEmbedding | None

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.in_project = nn.Linear(config.embedding_dim, 3 * config.num_attention_heads * config.attention_dim, bias=False)
        self.out_project = nn.Linear(config.num_attention_heads * config.attention_dim, config.embedding_dim, bias=False)
        self.num_attention_heads = config.num_attention_heads
        self.attention_dim = config.attention_dim
        self.embedding_dim = config.embedding_dim
        self.flash_attention = config.flash_attention
        self.rotary_embedding = config.rotary_embedding

        # TODO:debug and understand tf is this
        causal_mask = (torch.tril(torch.ones(config.sequence_length, config.sequence_length))
                       .view(1, 1, config.sequence_length, config.sequence_length))

        self.causal_mask = nn.Parameter(data=causal_mask, requires_grad=False)

    def forward(self, x: Float[Tensor, "BATCH_SIZE SEQUENCE_LENGTH EMBEDDING_DIM"]) -> Float[Tensor, "BATCH_SIZE SEQUENCE_LENGTH EMBEDDING_DIM"]:
        BATCH_SIZE, SEQUENCE_LENGTH, EMBEDDING_DIM = x.shape

        q: Float[Tensor, "BATCH_SIZE NUM_HEADS SEQUENCE_LENGTH ATTENTION_DIM"]
        k: Float[Tensor, "BATCH_SIZE NUM_HEADS SEQUENCE_LENGTH ATTENTION_DIM"]
        v: Float[Tensor, "BATCH_SIZE NUM_HEADS SEQUENCE_LENGTH ATTENTION_DIM"]

        q, k, v = self._project_input(x)

        if self.rotary_embedding:
            q = self.rotary_embedding.rotate_queries_or_keys(q)
            k = self.rotary_embedding.rotate_queries_or_keys(k)

        attention_output: Float[Tensor, "BATCH_SIZE NUM_HEADS SEQUENCE_LENGTH ATTENTION_DIM"] = (
            self._causal_attention(SEQUENCE_LENGTH, q, k, v) if not self.flash_attention
            else scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)
        )

        attention_output_reshaped: Float[Tensor, "BATCH_SIZE SEQUENCE_LENGTH NUM_HEADS*ATTENTION_DIM"] = (
            attention_output
            .transpose(2, 1)  # shift NUM_HEADS <-> SEQUENCE_LENGTH
            .contiguous()  # for optimization?
            .view(BATCH_SIZE, SEQUENCE_LENGTH, self.num_attention_heads * self.attention_dim)  # Reshape to stack head outputs side by side
        )

        output_projection: Float[Tensor, "BATCH_SIZE SEQUENCE_LENGTH EMBEDDING_DIM"] = (
            self._project_output(attention_output_reshaped)
        )

        return output_projection

    def _causal_attention(self,
                          sequence_length: int,
                          q: Float[Tensor, "BATCH_SIZE NUM_HEADS SEQUENCE_LENGTH ATTENTION_DIM"],
                          k: Float[Tensor, "BATCH_SIZE NUM_HEADS SEQUENCE_LENGTH ATTENTION_DIM"],
                          v: Float[Tensor, "BATCH_SIZE NUM_HEADS SEQUENCE_LENGTH ATTENTION_DIM"]) -> Float[Tensor, "BATCH_SIZE NUM_HEADS SEQUENCE_LENGTH ATTENTION_DIM"]:
        k_transposed: Float[Tensor, "BATCH_SIZE NUM_HEADS ATTENTION_DIM SEQUENCE_LENGTH"] = k.transpose(-2, -1)  # Align dimensions for dot product

        attention_scores: Float[Tensor, "BATCH_SIZE NUM_HEADS SEQUENCE_LENGTH SEQUENCE_LENGTH"] = (
                (q @ k_transposed) * (1.0 / math.sqrt(k.size(-1)))  # Scaled attention scores
        )

        masked_attention_scores: Float[Tensor, "BATCH_SIZE NUM_HEADS SEQUENCE_LENGTH SEQUENCE_LENGTH"] = (
            attention_scores.masked_fill(self.causal_mask[:, :, :sequence_length, :sequence_length] == 0, FLOAT_MIN)  # Apply causal mask
        )

        attention_weights: Float[Tensor, "BATCH_SIZE NUM_HEADS SEQUENCE_LENGTH SEQUENCE_LENGTH"] = (
            softmax(masked_attention_scores, dim=-1)
        )

        attention_output: Float[Tensor, "BATCH_SIZE NUM_HEADS SEQUENCE_LENGTH ATTENTION_DIM"] = (
                attention_weights @ v
        )

        return attention_output

    def _project_input(self, x: Float[Tensor, "BATCH_SIZE SEQUENCE_LENGTH EMBEDDING_DIM"]) -> Tuple[
        Float[Tensor, "BATCH_SIZE NUM_HEADS SEQUENCE_LENGTH ATTENTION_DIM"],
        Float[Tensor, "BATCH_SIZE NUM_HEADS SEQUENCE_LENGTH ATTENTION_DIM"],
        Float[Tensor, "BATCH_SIZE NUM_HEADS SEQUENCE_LENGTH ATTENTION_DIM"]
    ]:
        BATCH_SIZE, SEQUENCE_LENGTH, _ = x.shape

        input_projection: Float[Tensor, "BATCH_SIZE SEQUENCE_LENGTH 3*ATTENTION_DIM*NUM_HEADS"] = self.in_project(x)

        q, k, v = input_projection.split(self.attention_dim * self.num_attention_heads, dim=2)
        q: Float[Tensor, "BATCH_SIZE NUM_HEADS SEQUENCE_LENGTH ATTENTION_DIM"] = (
            q.view(BATCH_SIZE, SEQUENCE_LENGTH, self.num_attention_heads, self.attention_dim).transpose(1, 2)
        )
        k: Float[Tensor, "BATCH_SIZE NUM_HEADS SEQUENCE_LENGTH ATTENTION_DIM"] = (
            k.view(BATCH_SIZE, SEQUENCE_LENGTH, self.num_attention_heads, self.attention_dim).transpose(1, 2)
        )
        v: Float[Tensor, "BATCH_SIZE NUM_HEADS SEQUENCE_LENGTH ATTENTION_DIM"] = (
            v.view(BATCH_SIZE, SEQUENCE_LENGTH, self.num_attention_heads, self.attention_dim).transpose(1, 2)
        )

        return q, k, v

    def _project_output(self, x: Float[Tensor, "BATCH_SIZE SEQUENCE_LENGTH NUM_HEADS*ATTENTION_DIM"]) -> Float[Tensor, "BATCH_SIZE SEQUENCE_LENGTH EMBEDDING_DIM"]:
        output_projection: Float[Tensor, "BATCH_SIZE SEQUENCE_LENGTH EMBEDDING_DIM"] = (
            self.out_project(x)
        )

        return output_projection


class TransformerBlock(nn.Module):
    input_norm: LayerNorm
    attention_output_norm: LayerNorm
    attention: MultiHeadedAttention
    feed_forward: FFN

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.input_norm = LayerNorm(config.embedding_dim)
        self.attention_output_norm = LayerNorm(config.embedding_dim)
        self.attention = MultiHeadedAttention(config)
        self.feed_forward = FFN(config)

    def forward(self, x: Float[Tensor, "BATCH_SIZE SEQUENCE_LENGTH EMBEDDING_DIM"]) -> Float[Tensor, "BATCH_SIZE SEQUENCE_LENGTH EMBEDDING_DIM"]:
        normalized_input = self.input_norm(x)
        attention_output = self.attention(normalized_input)
        normalized_attention_output = self.attention_output_norm(attention_output + x)
        feed_forward_output = self.feed_forward(normalized_attention_output)
        return feed_forward_output + x

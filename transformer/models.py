from typing import Tuple, Optional

import torch
from torch import nn, ModuleDict, Tensor
from torch.nn import ModuleList

from transformer.config import TransformerConfig
from transformer.layers import RotaryPositionalEmbedding, TransformerBlock, PositionalEmbedding


class GPT(nn.Module):
    modules: ModuleDict
    config: TransformerConfig

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        self.embeddings = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.positional_embedding = PositionalEmbedding(config)
        self.transformer_blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.block_num)])
        self.layer_norm = nn.LayerNorm(config.embedding_dim)
        self.language_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)

        self.embeddings.weight = self.language_head.weight

    def forward(self, x: Tensor, targets: Tensor = None) -> Tuple[Tensor, Optional[Tensor]]:
        device = x.device
        b, t = x.size()

        assert t <= self.config.sequence_length, f"Cannot forward sequence of length {t}, sequence length is only {self.config.sequence_length}"

        token_embeddings = self.embeddings(x)
        positional_embeddings = self.positional_embedding(token_embeddings)

        embedding_stream = positional_embeddings

        for block in self.transformer_blocks:
            embedding_stream = block(embedding_stream)

        normalized_embedding_stream = self.layer_norm(embedding_stream)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.language_head(normalized_embedding_stream)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.language_head(normalized_embedding_stream[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

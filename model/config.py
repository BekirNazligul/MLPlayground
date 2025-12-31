import dataclasses

from rotary_embedding_torch import RotaryEmbedding


@dataclasses.dataclass
class TransformerConfig:
    embedding_dim: int
    sequence_length: int
    num_attention_heads: int
    vocab_size: int
    block_num: int
    # key, query and value have same dim
    attention_dim: int
    feed_forward_dim: int
    flash_attention: bool = False
    rotary_embedding: RotaryEmbedding | None = None


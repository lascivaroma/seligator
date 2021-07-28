from typing import Dict, Tuple

import torch
import torch.nn as nn

from .attention import LinearAttentionWithoutQuery, LinearAttentionWithQuery


class BasisCustLinear(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            meta_param_manager: "MetaParamManager",
            metadata_emb_dim: int = 64,
            num_bases: int = 3,
            key_query_size: int = 64,
            metadata_categories: Dict[str, int] = None
    ):
        """

        :param metadata_emb_dim: Dimension of each metadata registry
        :param metadata_categories: Dictionary of categories where key is the name and value is the size of the
        vocabulary
        :param num_bases: Number of bases
        :param key_query_size: Dimension of the normalized vector of the concat of each metadata embedding vector
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.meta_units: Dict[str, int] = metadata_categories or {}
        for meta_name, meta_vocab_dim in metadata_categories.items():
            setattr(self, "num_" + meta_name, meta_vocab_dim)
            setattr(self, meta_name, nn.Embedding(meta_vocab_dim, metadata_emb_dim))
            meta_param_manager.register("BasisCustLinear." + meta_name, getattr(self, meta_name).weight)
        self.P = nn.Sequential(
            # From MetaData to Query
            nn.Linear(metadata_emb_dim * len(metadata_categories), key_query_size),
            nn.Tanh(),
            # Calculate Weights of each Basis: Key & Query Inner-product
            nn.Linear(key_query_size, num_bases, bias=False),
            nn.Softmax(dim=1),
            # Weighted Sum of Bases
            nn.Linear(num_bases, input_dim * output_dim)
        )

    def forward(self, x: torch.Tensor, metadata_vector: Dict[str, torch.tensor]):
        """

        :param x: Output of the encoder (most probably hidden state)
        :param metadata_vector: Dictionary of Metadata key -> Metadata Values
        """
        # X: (BatchSize, InputDim)
        # Metadata Vectors: Dict[str, Tensor(BatchSize)]

        # embs: (BatchSize, BatchSize*MetadataEmbDim)
        embs = torch.cat([getattr(self, name)(idx) for name, idx in metadata_vector.items()], dim=1)
        # weight: (BatchSize, InputDim, InputDim*OutputDim)
        weight = self.P(embs).view(x.shape[0], self.input_dim, self.output_dim)
        # Out: (BatchSize, InputDim, OutputDim)
        out = torch.bmm(x.unsqueeze(dim=1), weight).squeeze(dim=1)
        return out


class BasicBias(nn.Module):
    def __init__(self, output_dim: int):
        super().__init__()
        self.b = nn.Parameter(torch.zeros((1, output_dim)))

    def forward(self, metadata_vector: Dict[str, torch.Tensor]):
        return self.b


class BasicAttention(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.attention = LinearAttentionWithoutQuery(
            encoder_dim=input_dim,
        )

    def forward(self, x, mask, metadata_vector: Dict[str, torch.Tensor]):
        return self.attention(x, mask=mask)[0]


class BasisCustAttention(nn.Module):
    """

    :param metadata_emb_dim: Dimension of each metadata registry
    :param metadata_categories: Dictionary of categories where key is the name and value is the size of the
    vocabulary
    :param num_bases: Number of bases
    :param key_query_size: Dimension of the normalized vector of the concat of each metadata embedding vector
    """
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            meta_param_manager: "MetaParamManager",
            metadata_emb_dim: int = 64,
            num_bases: int = 3,
            key_query_size: int = 64,
            metadata_categories: Dict[str, int] = None
    ):
        super().__init__()
        for meta_name, meta_vocab_dim in metadata_categories.items():
            setattr(self, "num_" + meta_name, meta_vocab_dim)
            setattr(self, meta_name, nn.Embedding(meta_vocab_dim, metadata_emb_dim))
            meta_param_manager.register("BasisCustAttention." + meta_name, getattr(self, meta_name).weight)
        self.P = nn.Sequential(
            # From MetaData to Query
            nn.Linear(metadata_emb_dim * len(metadata_categories), key_query_size),
            nn.Tanh(),
            # Calculate Weights of each Basis: Key & Query Inner-product
            nn.Linear(key_query_size, num_bases, bias=False),
            nn.Softmax(dim=1),
            # Weighted Sum of Bases
            nn.Linear(num_bases, input_dim)
        )
        self.attention = LinearAttentionWithQuery(encoder_dim=input_dim, query_dim=input_dim)

    def forward(self, x, mask, metadata_vector: Dict[str, torch.Tensor]):
        return self.attention(
            x,
            query=self.P(torch.cat([
                getattr(self, name)(idx)
                for name, idx in metadata_vector.items()], dim=1
            ).unsqueeze(dim=1).repeat(1, x.shape[1], 1)),
            mask=mask)[0]


class MetaParamManager:
    def __init__(self):
        self.meta_em = {}

    def state_dict(self):
        return self.meta_em

    def register(self, name, param):
        self.meta_em[name] = param


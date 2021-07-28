from typing import Dict, Tuple

import torch
import torch.nn as nn

from .attention import LinearAttentionWithoutQuery, LinearAttentionWithQuery
from seligator.common.params import BasisVectorConfiguration


class BasisCustLinear(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            basis_vector_configuration: BasisVectorConfiguration
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

        for meta_name, meta_vocab_dim in basis_vector_configuration.categories.items():
            setattr(self, "num_" + meta_name, meta_vocab_dim)
            setattr(self, meta_name, nn.Embedding(meta_vocab_dim, basis_vector_configuration.emb_dim))
            basis_vector_configuration.param_manager.register(
                "BasisCustLinear." + meta_name, getattr(self, meta_name).weight
            )

        self.P = nn.Sequential(
            # From MetaData to Query
            nn.Linear(
                basis_vector_configuration.emb_dim * len(basis_vector_configuration.categories),
                basis_vector_configuration.key_query_size
            ),
            nn.Tanh(),
            # Calculate Weights of each Basis: Key & Query Inner-product
            nn.Linear(basis_vector_configuration.key_query_size, basis_vector_configuration.num_bases, bias=False),
            nn.Softmax(dim=1),
            # Weighted Sum of Bases
            nn.Linear(basis_vector_configuration.num_bases, input_dim * output_dim)
        )

    def forward(self, x: torch.Tensor, metadata_vector: Dict[str, torch.tensor]):
        """

        :param x: Output of the encoder (most probably hidden state)
        :param metadata_vector: Dictionary of Metadata key -> Metadata Values
        """
        # X: (BatchSize, InputDim)
        # Metadata Vectors: Dict[str, Tensor(BatchSize)]

        # embs: (BatchSize, Categories*MetadataEmbDim)
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
        return self.attention(x, mask=mask)


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
            basis_vector_configuration: BasisVectorConfiguration
    ):
        super().__init__()
        for meta_name, meta_vocab_dim in basis_vector_configuration.categories.items():
            setattr(self, "num_" + meta_name, meta_vocab_dim)
            setattr(self, meta_name, nn.Embedding(meta_vocab_dim, basis_vector_configuration.emb_dim))
            basis_vector_configuration.param_manager.register(
                "BasisCustAttention." + meta_name, getattr(self, meta_name).weight
            )
        self.P = nn.Sequential(
            # From MetaData to Query
            nn.Linear(
                basis_vector_configuration.emb_dim * len(basis_vector_configuration.categories),
                basis_vector_configuration.key_query_size),
            nn.Tanh(),
            # Calculate Weights of each Basis: Key & Query Inner-product
            nn.Linear(basis_vector_configuration.key_query_size, basis_vector_configuration.num_bases, bias=False),
            nn.Softmax(dim=1),
            # Weighted Sum of Bases
            nn.Linear(basis_vector_configuration.num_bases, input_dim)
        )
        self.attention = LinearAttentionWithQuery(
            encoder_dim=input_dim,
            query_dim=input_dim
        )

    def forward(self, x, mask, metadata_vector: Dict[str, torch.Tensor]):
        # x: (BatchSize, SeqLen, RnnOut)

        # embs: (BatchSize, Categories*MetadataEmbDim)
        # eg 4 categories, 4 batch size, 64 emb_dim : (4,256)
        embs = torch.cat([getattr(self, name)(idx) for name, idx in metadata_vector.items()], dim=1)

        # embs: (BatchSize, SeqLen, RnnOut)
        embs = embs.unsqueeze(dim=1).repeat(1, x.shape[1], 1)

        # query = (BatchSize, SeqLen, RnnOut)
        query = self.P(embs)

        return self.attention(
            x,
            query=query,
            mask=mask
        )


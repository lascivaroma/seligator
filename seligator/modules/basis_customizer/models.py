from typing import Dict, Tuple

import torch
import torch.nn as nn


class BasisCustLinear(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            meta_param_manager: "MetaParamManager",
            metadata_emb_dim: int = 64,
            num_bases: int = 3,
            key_query_size: int = 64,
            meta_units: Tuple[str, ...] = None
    ):
        """

        :param metadata_emb_dim: Dimension of each metadata registry
        :param meta_units: List of Metadata category, eg. `("author", "citation-type")`. Order must be stable
        :param num_bases: Number of bases
        :param key_query_size: Dimension of the normalized vector of the concat of each metadata embedding vector
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.meta_units: Tuple[str, ...] = meta_units or {}

        for meta_idx, meta_name  in enumerate(meta_units):
            setattr(self, "num_" + meta_name, meta_idx)
            setattr(self, meta_name, nn.Embedding(meta_idx, metadata_emb_dim))
            meta_param_manager.register("BasisCustLinear." + meta_name, getattr(self, meta_name).weight)
        self.P = nn.Sequential(
            # From MetaData to Query
            nn.Linear(metadata_emb_dim * len(meta_units), key_query_size),
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
        embs = torch.cat([getattr(self, name)(idx) for name, idx in metadata_vector.items()])
        weight = self.P(embs, dim=1).view(x.shape[0], self.input_dim, self.output_dim)
        return torch.bmm(x.unsqueeze(dim=1), weight).squeeze(dim=1)


class BasicBias(nn.Module):
    def __init__(self, output_dim: int):
        super().__init__()
        self.b = nn.Parameter(torch.zeros((1, output_dim)))

    def forward(self, metadata_vector: Dict[str, torch.Tensor]):
        return self.b


class MetaParamManager:
    def __init__(self):
        self.meta_em = {}

    def state_dict(self):
        return self.meta_em

    def register(self, name, param):
        self.meta_em[name] = param


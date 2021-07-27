from typing import Dict, Optional, Tuple
import logging

import torch
import torch.nn as nn
from .models import BasisCustLinear, BasicBias, MetaParamManager

logger = logging.getLogger(__name__)


class MetadataEnrichedLinear(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 metadata_param_manager: Optional[MetaParamManager] = None,
                 metadata_emb_dim: int = 64,
                 num_bases: int = 3,
                 key_query_size: int = 64,
                 meta_units: Tuple[str, ...] = None
                 ):
        super(MetadataEnrichedLinear, self).__init__()
        self.linear = BasisCustLinear(
            input_dim=input_dim,
            output_dim=output_dim,
            meta_param_manager=metadata_param_manager if metadata_param_manager else MetaParamManager(),
            metadata_emb_dim=metadata_emb_dim,
            num_bases=num_bases,
            key_query_size=key_query_size,
            meta_units=meta_units
        )
        self.bias = BasicBias(output_dim=output_dim)
        self._output_dim: int = output_dim

    def get_input_dim(self) -> int:
        return self.linear.get_input_dim()

    def get_output_dim(self) -> int:
        return self._output_dim

    def forward(self, inputs, metadata_vectors: Dict[str, torch.Tensor]):
        x = self.linear(inputs)
        return x + self.bias(metadata_vectors)

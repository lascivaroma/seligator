from typing import Dict
import logging

import torch
import torch.nn as nn
from ...common.params import BasisVectorConfiguration
from .basis_customizer import BasisCustLinear, BasicBias

logger = logging.getLogger(__name__)


class MetadataEnrichedLinear(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 basis_vector_configuration: BasisVectorConfiguration
                 ):
        super(MetadataEnrichedLinear, self).__init__()
        self.linear = BasisCustLinear(
            input_dim=input_dim,
            output_dim=output_dim,
            basis_vector_configuration=basis_vector_configuration
        )
        self.bias = BasicBias(output_dim=output_dim)
        self._output_dim: int = output_dim

    def get_input_dim(self) -> int:
        return self.linear.get_input_dim()

    def get_output_dim(self) -> int:
        return self._output_dim

    def forward(self, inputs, metadata_vectors: Dict[str, torch.Tensor]):
        x = self.linear(inputs, metadata_vectors)
        return x + self.bias(metadata_vectors)

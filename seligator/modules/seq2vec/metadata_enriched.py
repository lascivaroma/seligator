from typing import Dict, Optional, Tuple
import logging

import torch
import torch.nn as nn
from ...common.params import BasisVectorConfiguration
from .basis_customizer.models import BasicAttention, BasisCustAttention, BasisCustBiLSTM
from .wrapper import ModifiedPytorchSeq2VecWrapper

logger = logging.getLogger(__name__)


class MetadataEnrichedAttentionalLSTM(nn.Module):
    with_attention = True
    use_metadata_vector = True

    def __init__(
            self,
            input_dim: int,
            hidden_size: int,
            use_metadata_lstm: bool = False,
            use_metadata_attention: bool = False,
            basis_vector_configuration: BasisVectorConfiguration = None
    ):
        super(MetadataEnrichedAttentionalLSTM, self).__init__()
        self._input_dim: int = input_dim
        self._hidden_dim: int = hidden_size

        if use_metadata_lstm:
            self.lstm = BasisCustBiLSTM(
                input_dim=input_dim,
                hidden_size=hidden_size,
                basis_vector_configuration=basis_vector_configuration
            )
        else:
            self.lstm = ModifiedPytorchSeq2VecWrapper(
                module=nn.LSTM(
                    input_size=input_dim,
                    hidden_size=hidden_size,
                    bidirectional=True,
                    batch_first=True
                )
            )

        if use_metadata_attention:
            self.attention = BasisCustAttention(
                input_dim=self.get_output_dim(),
                basis_vector_configuration=basis_vector_configuration
            )
        else:
            self.attention = BasicAttention(input_dim=self.get_output_dim())

    def get_input_dim(self) -> int:
        return self._input_dim

    def get_output_dim(self) -> int:
        return self._hidden_dim * 2

    def forward(
            self,
            inputs: torch.Tensor,
            mask=torch.Tensor,
            metadata_vector: Dict[str, torch.Tensor] = None
    ):
        if isinstance(self.lstm, BasisCustBiLSTM):
            x = self.lstm(inputs, mask=mask, metadata_vector=metadata_vector)
        else:
            x = self.lstm(inputs, mask=mask)
        # pass
        x, attention = self.attention(x, mask=mask.float(), metadata_vector=metadata_vector)
        return x, attention

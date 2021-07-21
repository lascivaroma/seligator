from typing import Tuple, Optional
from torch.nn import GRU, Parameter, functional as F, Linear, Dropout, LSTM
import torch

from allennlp.modules.seq2vec_encoders import Seq2VecEncoder


def matrix_mul(inputs: torch.Tensor, weight: torch.Tensor, bias: Optional[Parameter] = None) -> torch.Tensor:
    feature_list = []
    for feature in inputs:
        feature = torch.mm(feature, weight)
        if isinstance(bias, Parameter):
            feature = feature + bias.expand(feature.size()[0], bias.size()[1])
        feature = torch.tanh(feature).unsqueeze(0)
        feature_list.append(feature)
    return torch.cat(feature_list, 0).squeeze(dim=2)


def element_wise_mul(input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
    feature_list = []
    for feature_1, feature_2 in zip(input1, input2):
        feature_2 = feature_2.unsqueeze(1).expand_as(feature_1)
        feature = feature_1 * feature_2
        feature_list.append(feature.unsqueeze(0))
    output = torch.cat(feature_list, 0)
    return sum(output, 0).unsqueeze(0)


class HierarchicalAttentionalEncoder(Seq2VecEncoder):
    with_attention: bool = True

    def __init__(self, input_dim: int, hidden_size: int, **kwargs):
        super(HierarchicalAttentionalEncoder, self).__init__(**kwargs)

        self._input_dim: int = input_dim

        # From https://github.com/uvipen/Hierarchical-attention-networks-pytorch/blob/master/src/word_att_model.py

        self.gru = GRU(input_dim, hidden_size, bidirectional=True, dropout=0.3, batch_first=True)
        self.context = Parameter(torch.Tensor(2 * hidden_size, 1), requires_grad=True)
        self.dense = Linear(2*hidden_size, 2*hidden_size)
        self.dropout = Dropout(0.3)
        self._create_weights(mean=0.0, std=0.05)

        self._output_dim: int = 2 * hidden_size

    def _create_weights(self, mean=0.0, std=0.05):
        self.context.data.normal_(mean, std)

    def get_input_dim(self) -> int:
        return self._input_dim

    def get_output_dim(self) -> int:
        return self._output_dim

    def forward(self, inputs: torch.Tensor, mask: Optional[torch.Tensor] = None,
                **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        # Inputs: (batch, seq_len, embedding_len)
        # mask (initial): (batch, seq_len)
        # mask (after): (batch, seq_len, 1)
        if mask is not None:
            mask = mask.long().unsqueeze(dim=-1)
        else:
            mask = torch.ones(inputs.shape[:2]).bool().unsqueeze(dim=-1)
        # word_output: (batch_size , sentence_len, nb_dir*gru_size*nb_layer)
        word_output, word_hidden = self.gru(inputs)
        word_output = self.dropout(word_output)
        # attention: (batch_size, sentence_len, 2*gru_size)
        word_attention = torch.tanh(self.dense(word_output))
        # weights: batch_size, sentence_len, 1
        weights = torch.matmul(word_attention, self.context)
        # weights : (batch_size, sentence_len, 1)
        weights = F.softmax(weights, dim=1)

        # weights : (batch_size, sentence_len, 1)
        weights = torch.where(mask != 0, weights, torch.full_like(mask, 0, dtype=torch.float, device=weights.device))

        # weights : (batch_size, sentence_len, 1)
        weights = weights / (torch.sum(weights, dim=1).unsqueeze(1) + 1e-4)

        output = torch.sum((weights * word_output), dim=1)

        return output, weights.squeeze(2)


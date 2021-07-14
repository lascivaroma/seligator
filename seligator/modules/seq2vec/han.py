from typing import Tuple, Optional
from torch.nn import GRU, Parameter, functional as F, Linear
from torch import Tensor, mm, tanh, cat, sum

from allennlp.modules.seq2vec_encoders import Seq2VecEncoder


def matrix_mul(inputs: Tensor, weight: Tensor, bias: Optional[Parameter] = None) -> Tensor:
    feature_list = []
    for feature in inputs:
        feature = mm(feature, weight)
        if isinstance(bias, Parameter):
            feature = feature + bias.expand(feature.size()[0], bias.size()[1])
        feature = tanh(feature).unsqueeze(0)
        feature_list.append(feature)
    return cat(feature_list, 0).squeeze(dim=2)


def element_wise_mul(input1: Tensor, input2: Tensor) -> Tensor:
    feature_list = []
    for feature_1, feature_2 in zip(input1, input2):
        feature_2 = feature_2.unsqueeze(1).expand_as(feature_1)
        feature = feature_1 * feature_2
        feature_list.append(feature.unsqueeze(0))
    output = cat(feature_list, 0)
    return sum(output, 0).unsqueeze(0)


class HierarchicalAttentionalEncoder(Seq2VecEncoder):
    with_attention: bool = True

    def __init__(self, input_dim: int, hidden_size: int, **kwargs):
        super(HierarchicalAttentionalEncoder, self).__init__(**kwargs)

        self._input_dim: int = input_dim

        # From https://github.com/uvipen/Hierarchical-attention-networks-pytorch/blob/master/src/word_att_model.py

        self.context_weight = Parameter(Tensor(2 * hidden_size, 1))

        self.gru = GRU(input_dim, hidden_size, bidirectional=True)
        self.dense = Linear(2*hidden_size, 2*hidden_size)
        self._create_weights(mean=0.0, std=0.05)

        self._output_dim: int = 2 * hidden_size

    def _create_weights(self, mean=0.0, std=0.05):
        self.context_weight.data.normal_(mean, std)

    def get_input_dim(self) -> int:
        return self._input_dim

    def get_output_dim(self) -> int:
        return self._output_dim

    def forward(self, inputs, *args, **kwargs) -> Tuple[Tensor, Tensor]:
        # f_output (n_words, batch_size, dimension * hidden)
        # h_output (dimensions * n_layers, n_words, hidden_size)
        f_output, h_output = self.gru(inputs.transpose(1, 0))
        attention = self.dense(f_output)
        attention = matrix_mul(attention, self.context_weight).permute(1, 0)
        # (batch_size, n_words)
        attention = F.softmax(attention)

        # (batch_size, output_dim)
        output = element_wise_mul(f_output, attention.permute(1, 0)).transpose(0, 1).squeeze(dim=1)

        return output, attention


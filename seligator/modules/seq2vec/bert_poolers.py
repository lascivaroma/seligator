import torch.nn as nn
from allennlp.modules import Seq2VecEncoder
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder


class SumAndLinear(Seq2VecEncoder):  # Does not improve the output
    def __init__(self, input_dim):
        super(SumAndLinear, self).__init__()
        self.embedding = BagOfEmbeddingsEncoder(input_dim)
        self.linear = nn.Linear(input_dim, input_dim)
        self._input_dim = input_dim
        self._output_dim = input_dim

    def get_input_dim(self) -> int:
        return self._input_dim

    def get_output_dim(self) -> int:
        return self._output_dim

    def forward(self, tokens, mask=None):
        summed = self.embedding(tokens, mask=mask)
        masked = self.linear(summed)
        return masked


class PoolerHighway(Seq2VecEncoder):  # Does not improve the output
    def __init__(self, encoder: Seq2VecEncoder, output_dim: int):
        super(PoolerHighway, self).__init__()
        self._encoder = encoder
        self.linear = nn.Linear(encoder.get_output_dim(), output_dim)
        self._output_dim = output_dim

    def get_input_dim(self) -> int:
        return self._encoder.get_input_dim()

    def get_output_dim(self) -> int:
        return self._output_dim

    def forward(self, tokens, mask=None):
        summed = self._encoder(tokens, mask=mask)
        masked = self.linear(summed)
        return masked

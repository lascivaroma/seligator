import torch
from torch import nn

from allennlp.modules import Seq2VecEncoder
from allennlp.modules.seq2vec_encoders import GruSeq2VecEncoder
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder

from seligator.common.params import BertPoolerClass
from .han import HierarchicalAttentionalEncoder


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


# Initial code
# https://github.com/UKPLab/sentence-transformers/
#   blob/d7235076a663114c5267b093d5c28e1fc0272f76/sentence_transformers/models/Pooling.py
class CustomBertPooler(Seq2VecEncoder):
    """Performs pooling (max or mean) on the token embeddings.

    Using pooling, it generates from a variable sized sentence a fixed sized sentence embedding. This layer also allows
    to use the CLS token if it is returned by the underlying word embedding model.
    You can concatenate multiple poolings together.
    """
    def __init__(
            self,
            input_dim: int = 768,
            mode: BertPoolerClass = None,
            reduce_dim: int = 256
        ):
        super(CustomBertPooler, self).__init__()

        self._input_dim = input_dim
        self._output_dim = input_dim
        self.linear = None
        self.encoder = None
        if mode == BertPoolerClass.MEANMAX:
            self._output_dim *= 2
        elif mode == BertPoolerClass.CLS_Highway:
            self._output_dim = reduce_dim
            self.linear = nn.Linear(self._input_dim, reduce_dim)
        elif mode == BertPoolerClass.GRU:
            self.encoder = GruSeq2VecEncoder(
                input_dim,
                hidden_size=reduce_dim//2,
                bidirectional=True
            )
            self._output_dim = self.encoder.get_output_dim()
        elif mode == BertPoolerClass.HAN:
            self.encoder = HierarchicalAttentionalEncoder(
                input_dim,
                hidden_size=reduce_dim//2
            )
            self._output_dim = self.encoder.get_output_dim()
        self.mode = mode
        self.dropout = nn.Dropout(.3)

    def get_output_dim(self) -> int:
        return self._output_dim

    def __repr__(self):
        return f"<CustomBertPooling mode={self.mode} />"

    def forward(self, tokens: torch.Tensor, mask: torch.BoolTensor = None):
        """

        tokens: batch * seq len * emb len
        """
        if self.mode in {BertPoolerClass.GRU, BertPoolerClass.HAN}:
            attention = None
            encoded = self.encoder(tokens, mask)
            if isinstance(encoded, tuple):
                encoded, attention = encoded
                encoded = self.dropout(encoded)
                return encoded, attention
            return encoded

        max_over_time, avgs = None, None
        if self.mode == BertPoolerClass.CLS or self.mode == BertPoolerClass.CLS_Highway:
            embs = tokens[:, 0, :]
            if self.mode == BertPoolerClass.CLS_Highway:
                embs = self.linear(embs)
            return self.dropout(embs)

        if self.mode in {BertPoolerClass.MEANMAX, BertPoolerClass.MAX}:
            tokens[mask == False] = -1e9  # Set padding tokens to large negative value
            max_over_time = torch.max(tokens, 1)[0]
            if self.mode == BertPoolerClass.MAX:
                return self.dropout(max_over_time)

        if self.mode in {BertPoolerClass.MEANMAX, BertPoolerClass.MEAN}:
            input_mask_expanded = mask.unsqueeze(-1).expand(tokens.size()).float()
            sum_embeddings = torch.sum(tokens * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            avgs = sum_embeddings / sum_mask
            if self.mode == BertPoolerClass.MEAN:
                return self.dropout(avgs)
        # if we are still there, we MEANMAX
        return self.dropout(torch.cat([avgs, max_over_time], dim=-1))

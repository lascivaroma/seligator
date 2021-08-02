from typing import Dict

import torch
import torch.nn as nn

from allennlp.nn.util import get_lengths_from_binary_sequence_mask

from .attention import LinearAttentionWithoutQuery, LinearAttentionWithQuery
from seligator.common.params import BasisVectorConfiguration


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


class BasisCustBiLSTM(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_size: int,
            basis_vector_configuration: BasisVectorConfiguration
    ):
        super().__init__()
        self._bvc = basis_vector_configuration
        self.num_bases = self._bvc.num_bases
        self.batch_first = True

        # ? I believe the original state_size is meant as the output (given how it's reused later)
        #  Except for us, this is the real hidden size. We might want to * 2 this value actually
        self.each_state = hidden_size
        self.hidden_size = hidden_size * 2
        self.word_dim = input_dim
        self.bidirectional: bool = True
        for name, num_meta in self._bvc.categories.items():
            setattr(self, "num_" + name, num_meta)
            setattr(self, name, nn.Embedding(num_meta, self._bvc.emb_dim))
            self._bvc.param_manager.register("BasisCustBiLSTM." + name, getattr(self, name).weight)

        self.weight_ih_l0 = nn.Parameter(torch.zeros(self._bvc.num_bases, self.hidden_size * 2, self.word_dim))
        self.weight_hh_l0 = nn.Parameter(torch.zeros(self._bvc.num_bases, self.hidden_size * 2, self.hidden_size//2))
        self.bias_l0 = nn.Parameter(torch.zeros(self._bvc.num_bases, self.hidden_size * 2))
        self.weight_ih_l0_reverse = nn.Parameter(torch.zeros(self._bvc.num_bases, self.hidden_size * 2, self.word_dim))
        self.weight_hh_l0_reverse = nn.Parameter(torch.zeros(self._bvc.num_bases, self.hidden_size * 2, self.hidden_size//2))
        self.bias_l0_reverse = nn.Parameter(torch.zeros(self._bvc.num_bases, self.hidden_size * 2))

        self.P = nn.Sequential(
            # From MetaData to Query
            nn.Linear(self._bvc.emb_dim * len(self._bvc.categories_tuple), self._bvc.key_query_size),
            nn.Tanh(),
            nn.Linear(self._bvc.key_query_size, self._bvc.num_bases, bias=False),
            # Calculate Weights of each Basis: Key & Query Inner-product
            nn.Softmax(dim=1),
        )

    def get_output_dim(self) -> int:
        return self.hidden_size  # Because BiDir

    def forward(self, x, mask, metadata_vector: Dict[str, torch.Tensor]):
        # length: Tensor(batchSize)
        length = get_lengths_from_binary_sequence_mask(mask)
        if len(metadata_vector) == 0:
            raise ValueError("There are no metadata data passed to the MetadataEnrichedLSTM")
        query = torch.cat(
            [
                getattr(self, name)(idx)
                for name, idx in metadata_vector.items()
            ],
            dim=1
        )
        # c_batch:  Tensor (NumbCategory, NumBases)
        c_batch = self.P(query)
        num_bases = self.num_bases
        cell_size = self.each_state
        input_size = self.word_dim

        batch_size = x.size(0)
        maxlength = torch.max(length).item()

        # x_reverse is gonna be the same text but backward ! Hence reversing the IDX
        reverse_idx = torch.arange(maxlength - 1, -1, -1).to(x.device)
        # reverse_idx = torch.from_numpy(reverse_idx)
        #   -> Tensor(BatchSize, SequenceLen, EmbDim)
        x_reverse = x[:, reverse_idx, :]

        # First Operation is 4*4096
        weight_ih_l0 = torch.mm(c_batch, self.weight_ih_l0.view(num_bases, -1)).view(
            batch_size, cell_size * 4, input_size
        )  # batch_size, cell_size*4, input_size
        weight_hh_l0 = torch.mm(c_batch, self.weight_hh_l0.view(num_bases, -1)).view(batch_size, cell_size * 4,
                                                                                     cell_size)  # batch_size, cell_size*4, cell_size
        bias_l0 = torch.mm(c_batch, self.bias_l0)  # batch_size, cell_size*4
        weight_ih_l0_reverse = torch.mm(c_batch, self.weight_ih_l0_reverse.view(num_bases, -1)).view(batch_size,
                                                                                                     cell_size * 4,
                                                                                                     input_size)  # batch_size, cell_size*4, input_size
        weight_hh_l0_reverse = torch.mm(c_batch, self.weight_hh_l0_reverse.view(num_bases, -1)).view(batch_size,
                                                                                                     cell_size * 4,
                                                                                                     cell_size)  # batch_size, cell_size*4, cell_size
        bias_l0_reverse = torch.mm(c_batch, self.bias_l0_reverse)  # batch_size, cell_size*4

        (h0, c0) = torch.zeros((2, batch_size, cell_size, 1)).to(x.device)  # only for forward path
        (h0_reverse, c0_reverse) = torch.zeros((2, batch_size, cell_size, 1)).to(x.device)  # only for forward path
        hidden = (h0, c0)
        hidden_reverse = (h0_reverse, c0_reverse)
        htops = None
        htops_reverse = None
        for i in range(maxlength):
            hx, cx = hidden  # batch_size, cell_size, 1
            ix = x[:, i, :]  # batch_size, input_size
            ix = ix.unsqueeze(dim=2)  # batch_size, input_size, 1

            i2h = torch.bmm(weight_ih_l0, ix)
            i2h = i2h.squeeze(dim=2)  # batch_size, cell_size*4
            h2h = torch.bmm(weight_hh_l0, hx)
            h2h = h2h.squeeze(dim=2)  # batch_size, cell_size*4

            gates = i2h + h2h + bias_l0  # batch_size, cell_size*4
            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            cellgate = torch.tanh(cellgate)  # o_t
            outgate = torch.sigmoid(outgate)

            cx = cx.squeeze(dim=2)  # batch_size, cell_size
            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * torch.tanh(cy)  # batch_size, cell_size

            mask = (length - 1) < i
            if mask.sum() > 0:
                cy[mask] = torch.zeros(mask.sum(), cell_size).to(x.device)
                hy[mask] = torch.zeros(mask.sum(), cell_size).to(x.device)

            if (htops is None):
                htops = hy.unsqueeze(dim=1)
            else:
                htops = torch.cat((htops, hy.unsqueeze(dim=1)), dim=1)

            cx = cy.unsqueeze(dim=2)
            hx = hy.unsqueeze(dim=2)
            hidden = (hx, cx)

            ###############################################################################

            # reverse
            hx_reverse, cx_reverse = hidden_reverse  # batch_size, cell_size, 1
            ix_reverse = x_reverse[:, i, :]  # batch_size, input_size
            ix_reverse = ix_reverse.unsqueeze(dim=2)  # batch_size, input_size, 1

            i2h = torch.bmm(weight_ih_l0_reverse, ix_reverse)
            i2h = i2h.squeeze(dim=2)  # batch_size, cell_size*4
            h2h = torch.bmm(weight_hh_l0_reverse, hx_reverse)
            h2h = h2h.squeeze(dim=2)  # batch_size, cell_size*4

            gates = i2h + h2h + bias_l0_reverse  # batch_size, cell_size*4
            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            cellgate = torch.tanh(cellgate)  # o_t
            outgate = torch.sigmoid(outgate)

            cx_reverse = cx_reverse.squeeze(dim=2)  # batch_size, cell_size
            cy_reverse = (forgetgate * cx_reverse) + (ingate * cellgate)
            hy_reverse = outgate * torch.tanh(cy_reverse)  # batch_size, cell_size

            # mask
            mask_reverse = (maxlength - i) > length
            # mask_reverse = np.nonzero(mask_reverse)[0]
            # mask_reverse = torch.from_numpy(mask_reverse).to(self.device)
            if mask_reverse.sum() > 0:
                cy_reverse[mask_reverse] = torch.zeros(mask_reverse.sum(), cell_size).to(x.device)
                hy_reverse[mask_reverse] = torch.zeros(mask_reverse.sum(), cell_size).to(x.device)

            if (htops_reverse is None):
                htops_reverse = hy_reverse.unsqueeze(dim=1)
            else:
                htops_reverse = torch.cat((htops_reverse, hy_reverse.unsqueeze(dim=1)), dim=1)

            cx_reverse = cy_reverse.unsqueeze(dim=2)
            hx_reverse = hy_reverse.unsqueeze(dim=2)
            hidden_reverse = (hx_reverse, cx_reverse)

        # reverse order of backward batch
        reverse_idx = torch.arange(maxlength - 1, -1, -1).to(x.device)
        # reverse_idx = torch.from_numpy(reverse_idx).to(self.device)
        htops_reverse = htops_reverse[:, reverse_idx, :]

        # concatenate forward and backward path
        hiddens = torch.cat((htops, htops_reverse), dim=2)
        return hiddens

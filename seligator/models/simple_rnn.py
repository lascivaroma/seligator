from typing import Dict, Tuple, Optional

import torch.nn.functional
from allennlp.data import Vocabulary, TextFieldTensors
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2vec_encoders import LstmSeq2VecEncoder
from allennlp.nn import util
from allennlp.nn.util import get_text_field_mask

from seligator.models.base import BaseModel
from seligator.common.constants import EMBEDDING_DIMENSIONS


class SimpleRNNClassifier(BaseModel):
    def __init__(self,
                 vocab: Vocabulary,
                 input_feature_names: Tuple[str, ...],
                 embedder: TextFieldEmbedder,
                 encoder: LstmSeq2VecEncoder,
                 emb_dropout: float = 0.3,
                 **kwargs):
        super().__init__(vocab, input_feature_names=input_feature_names)

        self.embedder = embedder
        self.encoder = encoder
        self.classifier = torch.nn.Linear(
            encoder.get_output_dim(),
            self.num_labels
        )
        if emb_dropout:
            self._emb_dropout = torch.nn.Dropout(emb_dropout)
        else:
            self._emb_dropout = None

    def forward(self,
                token: TextFieldTensors,
                label: Optional[torch.Tensor] = None,
                **tasks) -> Dict[str, torch.Tensor]:

        token = {
            cat: token[cat]
            for cat in self.input_feature_names
        }
        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.embedder(token)

        if self._emb_dropout is not None:
            embedded_text = self._emb_dropout(embedded_text)

        # Shape: (batch_size, num_tokens)
        mask = util.get_text_field_mask(token)

        # Shape: (batch_size, encoding_dim)
        encoded_text = self.encoder(embedded_text, mask)

        # Shape: (batch_size, num_labels)
        logits = self.classifier(encoded_text)

        # Shape: (batch_size, num_labels)
        probs = torch.nn.functional.softmax(logits)
        # Shape: (1,)
        output = {"probs": probs}

        if label is not None:
            self._compute_metrics(logits, label, output)

        return output


def build_model(
        vocab: Vocabulary,
        emb_dims: Dict[str, int] = None,
        use_only: Tuple[str, ...] = ("token", )
) -> Model:
    emb_dims = emb_dims or EMBEDDING_DIMENSIONS

    embedder = BasicTextFieldEmbedder(
        {
            cat: Embedding(embedding_dim=emb_dims[cat], num_embeddings=vocab.get_vocab_size(cat))
            for cat in use_only
        }

    )
    encoder = LstmSeq2VecEncoder(
        input_size=sum([emb_dims[cat] for cat in use_only]),
        hidden_size=300,
        num_layers=1,
        bidirectional=True,
        dropout=0.7
    )

    return SimpleRNNClassifier(
        vocab=vocab,
        embedder=embedder,
        encoder=encoder,
        input_feature_names=use_only
    )


if __name__ == "__main__":
    from seligator.training.trainer import run_training_loop
    model, dataset_reader = run_training_loop(
        build_model=build_model,
        cuda_device=-1,
        use_only=("token", )
    )

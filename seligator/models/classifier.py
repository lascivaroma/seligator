from typing import Dict, Tuple, Optional, Callable

import torch.nn.functional
from allennlp.data import Vocabulary, TextFieldTensors
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding, TokenCharactersEncoder
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder
from allennlp.nn import util

from seligator.models.base import BaseModel
from seligator.common.constants import EMBEDDING_DIMENSIONS


class FeatureEmbeddingClassifier(BaseModel):
    def __init__(self,
                 vocab: Vocabulary,
                 input_features: Tuple[str, ...],
                 features_embedder: TextFieldEmbedder,
                 features_encoder: Seq2VecEncoder,
                 emb_dropout: float = 0.3,
                 **kwargs):
        super().__init__(vocab, input_features=input_features)

        if emb_dropout:
            self._emb_dropout = torch.nn.Dropout(emb_dropout)
        else:
            self._emb_dropout = None

        self.features_embedder = features_embedder
        self.features_encoder = features_encoder
        self.classifier = torch.nn.Linear(features_encoder.get_output_dim(), self.num_labels)

    def forward(self,
                label: Optional[torch.Tensor] = None,
                **inputs) -> Dict[str, torch.Tensor]:
        token = self._rebuild_input(inputs)

        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.features_embedder(token)

        if self._emb_dropout is not None:
            # Shape: (batch_size, num_tokens, embedding_dim)
            embedded_text = self._emb_dropout(embedded_text)

        # Shape: (batch_size, num_tokens)
        mask = util.get_text_field_mask(token)

        # Shape: (batch_size, encoding_dim)
        encoded_text = self.features_encoder(embedded_text, mask)

        # Shape: (batch_size, num_labels)
        logits = self.classifier(encoded_text)

        # Shape: (batch_size, num_labels)
        probs = torch.nn.functional.softmax(logits)

        # Shape: (1,)
        output = {"probs": probs}
        if label is not None:
            self._compute_metrics(logits, label, output)

        return output

    @staticmethod
    def build_embeddings(
            vocabulary: Vocabulary,
            input_features: Tuple[str, ...],
            emb_dims: Dict[str, int] = None,
            char_encoders: Dict[str, Seq2VecEncoder] = None
    ) -> BasicTextFieldEmbedder:
        emb_dims = emb_dims or EMBEDDING_DIMENSIONS
        emb = {
            cat: Embedding(embedding_dim=emb_dims[cat], num_embeddings=vocabulary.get_vocab_size(cat))
            for cat in input_features
            if "_subword" not in cat and "_char" not in cat
        }
        if char_encoders:
            emb.update({
                cat: TokenCharactersEncoder(
                    embedding=Embedding(
                        embedding_dim=emb_dims[cat],
                        num_embeddings=vocabulary.get_vocab_size(cat)
                    ),
                    encoder=char_encoders[cat],
                    dropout=0.3
                )
                for cat in input_features
                if "_char" in cat
            })
        return BasicTextFieldEmbedder(emb)

    @classmethod
    def build_model(
            cls,
            vocabulary: Vocabulary,
            emb_dims: Dict[str, int] = None,
            input_features: Tuple[str, ...] = ("token",),
            features_encoder: Callable[[int], Seq2VecEncoder] = BagOfEmbeddingsEncoder,
            char_encoders: Dict[str, Seq2VecEncoder] = None
    ) -> Model:
        emb = cls.build_embeddings(
            vocabulary=vocabulary,
            input_features=input_features,
            emb_dims=emb_dims,
            char_encoders=char_encoders
        )
        return cls(
            vocab=vocabulary,
            input_features=input_features,
            features_embedder=emb,
            features_encoder=features_encoder(emb.get_output_dim())
        )


if __name__ == "__main__":
    from allennlp.modules.seq2vec_encoders import LstmSeq2VecEncoder
    from seligator.training.trainer import generate_all_data, train_model

    # For test, just change the input feature here
    INPUT_FEATURES = ("token", "lemma", "token_char")

    train, dev, vocab, reader = generate_all_data(input_features=INPUT_FEATURES)

    embedding_encoders = {
        cat: LstmSeq2VecEncoder(
            input_size=EMBEDDING_DIMENSIONS[cat],
            hidden_size=EMBEDDING_DIMENSIONS[f"{cat}_encoded"],
            num_layers=2,
            bidirectional=True,
            dropout=.3
        )
        for cat in {"token_char", "lemma_char"}
    }

    model = FeatureEmbeddingClassifier.build_model(
        vocabulary=vocab,
        emb_dims=EMBEDDING_DIMENSIONS,
        input_features=INPUT_FEATURES,
        features_encoder=lambda input_dim: LstmSeq2VecEncoder(
            input_size=input_dim,
            hidden_size=128,
            dropout=0.3,
            bidirectional=True,
            num_layers=2
        ),
        char_encoders=embedding_encoders
    )

    model.cuda()


    train_model(
        model=model,
        train_loader=train,
        dev_loader=dev,
        cuda_device=0
    )

from typing import Tuple, Optional, Dict, Any, Callable, Union
from copy import deepcopy

import torch
import torch.nn as nn

from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder, TextFieldEmbedder
from allennlp.modules.token_embedders import Embedding, TokenCharactersEncoder
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder, Seq2VecEncoder

from allennlp.nn import util

from seligator.modules.embedders.latinBert import PretrainedTransformerEmbedder
from seligator.common.constants import EMBEDDING_DIMENSIONS


class MixedEmbeddingEncoder(nn.Module):
    """ Embbeds and encodes mixed input with multi-layer information (token, lemma, pos, etc.)

    :param input_features: List of features that are used
    """

    def __init__(self,
                 input_features: Tuple[str, ...],

                 # Normal input
                 use_features: bool = True,
                 features_embedder: Optional[TextFieldEmbedder] = None,
                 features_encoder: Optional[Seq2VecEncoder] = None,

                 # Bert Input
                 use_bert: bool = False,
                 bert_embedder: Optional[PretrainedTransformerEmbedder] = None,
                 bert_pooler: Optional[Seq2VecEncoder] = None,

                 mixer: Union[str, Callable] = "concat",
                 emb_dropout: Union[float, nn.Dropout] = 0.3
                 ):
        super().__init__()
        self.input_features = input_features

        if isinstance(emb_dropout, nn.Dropout):
            self._emb_dropout = emb_dropout
        elif emb_dropout:
            self._emb_dropout = nn.Dropout(emb_dropout)
        else:
            self._emb_dropout = None

        self.use_features: bool = use_features
        self.features_embedder = features_embedder
        self.features_encoder = features_encoder

        self.use_bert = use_bert
        self.bert_embedder: Optional[PretrainedTransformerEmbedder] = bert_embedder
        self.bert_pooler: Seq2VecEncoder = bert_pooler

        out_dim = 0
        self.mixer = None
        if use_features and use_bert:
            if mixer and not isinstance(mixer, str):
                self.mixer = mixer
            elif mixer == "concat":
                self.mixer = lambda inp1, inp2: torch.cat([inp1, inp2], -1)
                out_dim = self.features_encoder.get_output_dim() + self.bert_pooler.get_output_dim()
            elif mixer.startswith("Linear:"):
                out_dim = int(mixer.split(":")[-1])
                self._mixer = torch.nn.Linear(
                    self.features_encoder.get_output_dim() + self.bert_pooler.get_output_dim(),
                    out_dim
                )
                self.mixer = lambda inp1, inp2: self._mixer(torch.cat([inp1, inp2], -1))
        elif use_features:
            out_dim = self.features_encoder.get_output_dim()
        elif use_bert:
            out_dim = self.bert_pooler.get_output_dim()
        else:
            raise ValueError("Neither Bert or Features are used")
        self._out_dim = out_dim

    def get_output_dim(self) -> int:
        return self._out_dim

    def _rebuild_input(self, inputs: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, TextFieldTensors]:
        return {
                cat: inputs[cat][cat]
                for cat in self.input_features
                if not cat.endswith("_subword")
        }

    def _forward_bert(self, token):
        embedded = self.bert_embedder(token["token_ids"], mask=token["mask"])
        return self.bert_pooler(embedded, mask=token["mask"])

    def _forward_features(self, token) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        token = self._rebuild_input(token)

        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.features_embedder(token)
        if self._emb_dropout is not None:
            # Shape: (batch_size, num_tokens, embedding_dim)
            embedded_text = self._emb_dropout(embedded_text)
        # Shape: (batch_size, num_tokens)
        mask = util.get_text_field_mask(token)

        # Shape: (batch_size, encoding_dim)
        if getattr(self.features_encoder, "with_attention", False):
            return self.features_encoder(embedded_text, mask)
        else:
            return self.features_encoder(embedded_text, mask), None

    def forward(self, data) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """

        :param data: Input (TextFieldEmbedders)
        :returns: Encoded input + Dictionary of optional outputs

        """
        # Deal with Bert (key: token_subword)
        if self.use_bert and self.use_features:
            _bert = self._forward_bert(data["token_subword"]["token_subword"])
            _features, attention = self._forward_features(data)
            v = self.mixer(_bert, _features)
        elif self.use_bert:
            v = self._forward_bert(data["token_subword"]["token_subword"])
        elif self.use_features:
            v, attention = self._forward_features(data)
        else:
            raise ValueError("No features or bert used.")
        return v, {}

    @staticmethod
    def merge_default_embeddings(additional):
        return {
            **EMBEDDING_DIMENSIONS,
            **additional
        }

    @staticmethod
    def build_embeddings(
            vocabulary: Vocabulary,
            input_features: Tuple[str, ...],
            pretrained_embeddings: Optional[Dict[str, str]] = None,
            trainable_embeddings: Optional[Dict[str, str]] = None,
            emb_dims: Dict[str, int] = None,
            char_encoders: Dict[str, Seq2VecEncoder] = None
    ) -> BasicTextFieldEmbedder:
        emb_dims = emb_dims or EMBEDDING_DIMENSIONS

        def get_data(cat, dico: Optional[Dict[str, str]], default=None):
            if dico and cat in dico:
                return dico[cat]
            return default

        emb = {
            cat: Embedding(
                embedding_dim=emb_dims[cat],
                num_embeddings=vocabulary.get_vocab_size(cat),
                pretrained_file=get_data(cat, pretrained_embeddings),
                trainable=get_data(cat, trainable_embeddings, default=True),
                vocab=vocabulary,
                vocab_namespace=cat
            )
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
    def build(
            cls,
            vocabulary: Vocabulary,

            emb_dims: Dict[str, int] = None,
            input_features: Tuple[str, ...] = ("token",),
            pretrained_embeddings: Optional[Dict[str, str]] = None,
            trainable_embeddings: Optional[Dict[str, str]] = None,

            features_encoder: Callable[[int], Seq2VecEncoder] = BagOfEmbeddingsEncoder,
            char_encoders: Dict[str, Seq2VecEncoder] = None,

            use_bert: bool = False,
            bert_embedder: Optional[PretrainedTransformerEmbedder] = None,
            bert_pooler: Optional[Seq2VecEncoder] = None,

            mixer: str = "concat",
            emb_dropout: float = 0.3
    ) -> "MixedEmbeddingEncoder":
        emb = cls.build_embeddings(
            vocabulary,
            input_features=input_features,
            emb_dims=emb_dims or EMBEDDING_DIMENSIONS,
            char_encoders=char_encoders,
            pretrained_embeddings=pretrained_embeddings,
            trainable_embeddings=trainable_embeddings
        )
        return cls(
            input_features=input_features,

            use_features=len([x for x in input_features if not x.endswith("subword")]) >= 1,
            features_embedder=emb,
            features_encoder=features_encoder(emb.get_output_dim()),

            use_bert=use_bert,
            bert_embedder=bert_embedder,
            bert_pooler=bert_pooler,

            mixer=mixer,
            emb_dropout=emb_dropout
        )

    def copy_for_siamese(self, copy: bool = True) -> "MixedEmbeddingEncoder":
        return type(self)(
            input_features=self.input_features,

            use_features=self.use_features,
            features_embedder=self.features_embedder,
            features_encoder=self.features_encoder if not copy else deepcopy(self.features_encoder),

            use_bert=self.use_bert,
            bert_embedder=self.bert_embedder,
            bert_pooler=self.bert_pooler if not copy else deepcopy(self.bert_pooler),
            mixer=self.mixer,
            emb_dropout=self._emb_dropout
        )

    @classmethod
    def build_for_siamese(cls, copy: bool = True, **build_kwargs) -> Tuple["MixedEmbeddingEncoder", ...]:
        left = cls.build(*build_kwargs)
        right = left.copy_for_siamese(copy=copy)
        return left, right


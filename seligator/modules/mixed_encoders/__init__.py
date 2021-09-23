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
from seligator.modules.embedders.mixed import FeatureAndTextEmbedder
from seligator.common.constants import EMBEDDING_DIMENSIONS, MSD_CAT_NAME
from seligator.dataset.dataloader import get_vocabulary_from_pretrained_embedding


class RaiseDebug(Exception):
    def __init__(self, message, tokens):
        self.msg = message
        self.tokens = tokens


class MixedEmbeddingEncoder(nn.Module):
    """ Embbeds and encodes mixed input with multi-layer information (token, lemma, pos, etc.)

    :param input_features: List of features that are used
    """

    def __init__(self,
                 input_features: Tuple[str, ...],

                 # Normal input
                 use_features: bool = True,
                 features_embedder: Optional[Union[TextFieldEmbedder, FeatureAndTextEmbedder]] = None,
                 features_encoder: Optional[Seq2VecEncoder] = None,

                 # Bert Input
                 use_bert: bool = False,
                 bert_embedder: Optional[PretrainedTransformerEmbedder] = None,
                 bert_pooler: Optional[Seq2VecEncoder] = None,

                 mixer: Union[str, Callable] = "concat",
                 emb_dropout: Union[float, nn.Dropout] = 0.5,
                 return_bert: bool = False,
                 ):
        super().__init__()
        self.input_features = input_features

        if isinstance(emb_dropout, nn.Dropout):
            self._emb_dropout = emb_dropout
        elif emb_dropout:
            self._emb_dropout = nn.Dropout(emb_dropout)
        else:
            self._emb_dropout = None

        self.use_metadata_vector = getattr(features_encoder, "use_metadata_vector", False)

        self.use_features: bool = use_features
        self.features_embedder = features_embedder
        self.features_encoder = features_encoder

        self.use_bert = use_bert
        self.bert_embedder: Optional[PretrainedTransformerEmbedder] = bert_embedder
        self.bert_pooler: Seq2VecEncoder = bert_pooler

        out_dim = 0
        self.mixer = None
        if use_features and use_bert:
            # mixer can be "token-level"
            if mixer == "token-level":
                self.mixer = "token-level"
                out_dim = self.features_encoder.get_output_dim()
            elif mixer and not isinstance(mixer, str):
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
        self.return_bert: bool = return_bert

    def get_output_dim(self) -> int:
        return self._out_dim

    def _rebuild_input(self, inputs: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, TextFieldTensors]:
        return {
                cat: inputs[cat] if cat == MSD_CAT_NAME else inputs[cat][cat]
                for cat in self.input_features
                if not cat.endswith("_subword")
        }

    def _forward_bert(self, token) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        embedded = self.bert_embedder(token["token_ids"], mask=token["mask"])
        pooled = self.bert_pooler(embedded, mask=token["mask"])

        if isinstance(pooled, tuple):
            return pooled[0], {
                "attention": pooled[1],
                # "bert_projection": embedded.tolist()
            }
        if self.return_bert:
            return pooled, embedded.tolist()
        return pooled, None

    def _align_bert(self, bert_tokens: Dict[str, torch.Tensor]) -> torch.Tensor:
        embedded = self.bert_embedder(bert_tokens["token_ids"], mask=bert_tokens["mask"])
        out = torch.zeros(
            (embedded.shape[0], bert_tokens["orig_token_ids"][bert_tokens["mask"] == True].max()+1, embedded.shape[-1]),
            device=embedded.device
        )
        # We should probably actually set out to padding values but whatever
        # print(bert_tokens["orig_token_ids"][bert_tokens["mask"] == True].max(), max(list(range(bert_tokens["orig_token_ids"][bert_tokens["mask"] == True].max()))))
        # We avoid the CLS token
        for batch_idx, (emb, orig, mask) in enumerate(zip(
                embedded[:, 1:, :], bert_tokens["orig_token_ids"][:, 1:], bert_tokens["mask"][:, 1:]
        )):
            # Here we have a batch
            orig, emb = orig[mask == True], emb[mask == True]
            # print(orig.max()+1, orig.shape)
            for word_id in range(orig.max()+1):
                out[batch_idx, word_id] = torch.mean(emb[orig == word_id])
        # There might be a fast way to do this but I don't know how
        return out

    def _forward_merged_features_and_bert(
            self,
            tokens,
            metadata_vector: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        bert_text = self._align_bert(tokens["token_subword"]["token_subword"])

        token = self._rebuild_input(tokens)

        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.features_embedder(token)

        if self._emb_dropout is not None:
            # Shape: (batch_size, num_tokens, embedding_dim)
            embedded_text = self._emb_dropout(embedded_text)

        # Shape: (batch_size, num_tokens)
        mask = util.get_text_field_mask({
            tok_cat: token[tok_cat]
            for tok_cat in token
            if tok_cat != MSD_CAT_NAME
        })
        if embedded_text.shape[:-1] != bert_text.shape[:-1]:
            with open("runtime_error.json", "w") as f:
                f.write(str(tokens))
            raise RaiseDebug(
                "Bert and the original text are not the same",
                tokens=tokens["lemma"]["lemma"]["tokens"].tolist()
            )
        embed = torch.cat([embedded_text, bert_text], dim=-1)

        if hasattr(self.features_encoder, "use_metadata_vector"):
            out = self.features_encoder(embed, mask=mask, metadata_vector=metadata_vector)
        else:
            out = self.features_encoder(embed, mask)
        if isinstance(out, tuple):
            return out
        return out, None

    def _forward_features(self, token, metadata_vector: Dict[str, torch.Tensor]
                          ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        token = self._rebuild_input(token)

        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.features_embedder(token)
        if self._emb_dropout is not None:
            # Shape: (batch_size, num_tokens, embedding_dim)
            embedded_text = self._emb_dropout(embedded_text)

        # Shape: (batch_size, num_tokens)
        mask = util.get_text_field_mask({
            tok_cat: token[tok_cat]
            for tok_cat in token
            if tok_cat != MSD_CAT_NAME
        })

        if hasattr(self.features_encoder, "use_metadata_vector"):
            out = self.features_encoder(embedded_text, mask=mask, metadata_vector=metadata_vector)
        else:
            out = self.features_encoder(embedded_text, mask)
        if isinstance(out, tuple):
            return out
        return out, None

    def forward(self, data, metadata_vector: Dict[str, torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """

        :param data: Input (TextFieldEmbedders)
        :returns: Encoded input + Dictionary of optional outputs

        """
        # Deal with Bert (key: token_subword)
        attention = None
        additional_output = {}
        if self.use_bert and self.use_features and self.mixer == "token-level":
            raise ValueError("Because of CLTK used as tokenizer")
            v, attention = self._forward_merged_features_and_bert(
                data,
                metadata_vector=metadata_vector
            )
        elif self.use_bert and self.use_features:
            _bert, embedded = self._forward_bert(data["token_subword"]["token_subword"])
            if embedded is not None:
                additional_output["bert_projection"] = embedded
            _features, attention = self._forward_features(data, metadata_vector=metadata_vector)
            v = self.mixer(_bert, _features)
        elif self.use_bert:
            v, embedded = self._forward_bert(data["token_subword"]["token_subword"])
            if embedded is not None:
                if isinstance(embedded, dict):
                    additional_output.update({
                        key: val.tolist() if isinstance(val, torch.Tensor) else val
                        for key, val in embedded.items()
                    })
                else:
                    additional_output["bert_projection"] = embedded
        elif self.use_features:
            v, attention = self._forward_features(data, metadata_vector=metadata_vector)
        else:
            raise ValueError("No features or bert used.")
        return v, {
            "attention": attention.tolist() if attention is not None else [],
            "doc-vectors": v.tolist(),
            **additional_output
        }

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
            trainable_embeddings: Optional[Dict[str, bool]] = None,
            emb_dims: Dict[str, int] = None,
            char_encoders: Dict[str, Seq2VecEncoder] = None,
            keep_all_vocab: bool = True,

    ) -> Union[BasicTextFieldEmbedder, FeatureAndTextEmbedder]:
        emb_dims = emb_dims or EMBEDDING_DIMENSIONS

        def get_data(cat, dico: Optional[Dict[str, str]], default=None):
            if dico and cat in dico:
                return dico[cat]
            return default

        # get_vocabulary_from_pretrained_embedding
        if keep_all_vocab and pretrained_embeddings:
            for cat in pretrained_embeddings:
                tokens = list(get_vocabulary_from_pretrained_embedding(pretrained_embeddings[cat]))
                vocabulary.add_tokens_to_namespace(tokens, cat)

        # input_features
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
            if "_subword" not in cat and "_char" not in cat and cat != MSD_CAT_NAME
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

        if MSD_CAT_NAME in input_features:
            return FeatureAndTextEmbedder(
                text_embedder=BasicTextFieldEmbedder(emb),
                feature_embedder_in=vocabulary.get_vocab_size(MSD_CAT_NAME),
                feature_embedder_out=emb_dims[MSD_CAT_NAME]
            )
        return BasicTextFieldEmbedder(emb)

    @classmethod
    def build(
            cls,
            vocabulary: Vocabulary,

            emb_dims: Dict[str, int] = None,
            input_features: Tuple[str, ...] = ("token",),

            features_encoder: Callable[[int], Seq2VecEncoder] = BagOfEmbeddingsEncoder,
            char_encoders: Dict[str, Seq2VecEncoder] = None,

            use_bert: bool = False,
            bert_embedder: Optional[PretrainedTransformerEmbedder] = None,
            bert_pooler: Optional[Seq2VecEncoder] = None,

            mixer: str = "concat",
            emb_dropout: float = 0.3,
            model_embedding_kwargs=None
    ) -> "MixedEmbeddingEncoder":
        emb = cls.build_embeddings(
            vocabulary,
            input_features=input_features,
            emb_dims=emb_dims or EMBEDDING_DIMENSIONS,
            char_encoders=char_encoders,
            **(model_embedding_kwargs or {})
        )

        features_encoder_dim = emb.get_output_dim()
        if bert_embedder is not None and bert_pooler is None and mixer == "token-level":
            features_encoder_dim = emb.get_output_dim() + bert_embedder.get_output_dim()
        return cls(
            input_features=input_features,

            use_features=len([x for x in input_features if not x.endswith("subword")]) >= 1,
            features_embedder=emb,
            features_encoder=features_encoder(features_encoder_dim),

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


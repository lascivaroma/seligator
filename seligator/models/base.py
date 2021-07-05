from typing import Dict, List, Optional, Tuple

import torch.nn.functional
import torch.nn as nn
from allennlp.data import Vocabulary, TextFieldTensors
from allennlp.models import Model
from allennlp.training.metrics import CategoricalAccuracy, FBetaMeasure
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding, TokenCharactersEncoder
from allennlp.modules import Seq2VecEncoder


from seligator.common.constants import EMBEDDING_DIMENSIONS


class BaseModel(Model):

    BERT_COMPATIBLE: bool = False
    IS_SIAMESE: bool = False

    def __init__(self,
                 vocab: Vocabulary,
                 input_features: Tuple[str, ...],
                 **kwargs):
        super().__init__(vocab)

        self.input_features: Tuple[str, ...] = input_features

        self.num_labels = vocab.get_vocab_size("labels")
        self.labels = vocab.get_index_to_token_vocabulary("labels")
        self._accuracy = CategoricalAccuracy()
        self._measure = FBetaMeasure()
        self._measure_macro = FBetaMeasure(average="macro")
        self._loss = nn.CrossEntropyLoss()

    def _compute_metrics(self, logits, label, output):
        self._accuracy(logits, label)
        self._measure(logits, label)
        self._measure_macro(logits, label)
        # Shape: (1,)
        output['loss'] = self._loss(logits, label)

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        beta: Dict[str, List[float]] = self._measure.get_metric(reset)
        return {
            "accuracy": self._accuracy.get_metric(reset),
            **{
                key: score
                for key, score in self._measure_macro.get_metric(reset).items()
              },
            **{
                f"{key}-{self.labels[score_idx]}": score
                for key, scores in beta.items()
                for score_idx, score in enumerate(scores)
            }
        }

    def _get_metrics(self, name, metric, reset: bool = False) -> Dict[str, float]:
        metric_out = metric.get_metric(reset)
        if isinstance(metric_out, float):
            return {name: metric_out}
        else:
            return {
                key: score
                for key, score in metric_out.items()
            }

    def _rebuild_input(self, inputs: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, TextFieldTensors]:
        return {
                cat: inputs[cat][cat]
                for cat in self.input_features
                if not cat.endswith("_subword")
        }

    def forward(self,
                label: Optional[torch.Tensor] = None,
                **tasks) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

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
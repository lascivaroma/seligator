from typing import Dict, List, Optional, Tuple

import torch.nn.functional
import torch.nn as nn
from allennlp.data import Vocabulary, TextFieldTensors
from allennlp.models import Model
from allennlp.training.metrics import CategoricalAccuracy, FBetaMeasure


class BaseModel(Model):

    def __init__(self,
                 vocab: Vocabulary,
                 input_feature_names: Tuple[str, ...],
                 **kwargs):
        super().__init__(vocab)

        self.input_feature_names: Tuple[str, ...] = input_feature_names

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

    def forward(self,
                token: TextFieldTensors,
                label: Optional[torch.Tensor] = None,
                **tasks) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

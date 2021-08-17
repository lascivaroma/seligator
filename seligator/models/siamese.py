# https://github.com/PonteIneptique/jdnlp

import torch
import torch.nn.functional as F
import random

from typing import Dict, Optional, Tuple, List, Union, Any
from overrides import overrides
import logging

from allennlp.data import Vocabulary
from allennlp.modules import Seq2VecEncoder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import CategoricalAccuracy, BooleanAccuracy

from seligator.models.base import BaseModel
from seligator.modules.mixed_encoders import MixedEmbeddingEncoder
from pytorch_metric_learning.losses import ContrastiveLoss
from pytorch_metric_learning.miners import BatchEasyHardMiner
from seligator.common.params import get_metadata_field_name, BasisVectorConfiguration

logger = logging.getLogger(__name__)


_Fields = Dict[str, torch.LongTensor]


def get_tensor_dict_subset(tensor_dict: Dict[str, Any], indices: torch.Tensor):
    def map_value(value):
        if isinstance(value, dict):
            return get_tensor_dict_subset(value, indices)
        elif isinstance(value, list):
            return [value[index] for index in indices.tolist()]
        elif isinstance(value, torch.Tensor):
            return value[indices]
        else:
            print(type(value))
            return value

    return dict(**{
        key: map_value(array)
        for key, array in tensor_dict.items()
    })


class SiameseClassifier(BaseModel):
    BERT_COMPATIBLE: bool = True
    IS_SIAMESE: bool = True
    INSTANCE_TYPE: str = "siamese"

    def __init__(self,
                 vocab: Vocabulary,
                 input_features: Tuple[str, ...],
                 mixed_encoder: Union[MixedEmbeddingEncoder, Tuple[MixedEmbeddingEncoder, MixedEmbeddingEncoder]],

                 loss_margin: float = 0.3,
                 prediction_threshold: float = 0.3,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 basis_vector_configuration: BasisVectorConfiguration = None,
                 loss: str = "contrastive",
                 **kwargs):
        super().__init__(vocab, input_features=input_features)

        self._reader_mode = kwargs.get("reader_mode", "default")
        self._miner = None
        if self._reader_mode not in {"default", "miner"}:
            raise ValueError("ReaderMode can only be `default` or `miner`")
        elif self._reader_mode == "miner":
            self._miner = BatchEasyHardMiner()
            self._all_miner = BatchEasyHardMiner(
                neg_strategy=BatchEasyHardMiner.ALL,
                pos_strategy=BatchEasyHardMiner.ALL
            )
        self.prediction_threshold = prediction_threshold
        self.regularizer: Optional[RegularizerApplicator] = regularizer

        self.left_encoder: Optional[Seq2VecEncoder] = None
        self.right_encoder: Optional[Seq2VecEncoder] = None

        self.metadata_categories: Tuple[str, ...] = basis_vector_configuration.categories_tuple

        if isinstance(mixed_encoder, MixedEmbeddingEncoder):
            self.left_encoder = mixed_encoder
            self.right_encoder = self.left_encoder
        elif isinstance(mixed_encoder, tuple) and \
                isinstance(mixed_encoder[0], MixedEmbeddingEncoder) and\
                isinstance(mixed_encoder[1], MixedEmbeddingEncoder):
            self.left_encoder = mixed_encoder[0]
            self.right_encoder = mixed_encoder[1]

        self.metrics = {
            "sim-accuracy": BooleanAccuracy(),
            "accuracy": CategoricalAccuracy(),
            #"f1": FBetaMeasure()
        }
        if loss.lower() == "contrastive":
            self._loss = ContrastiveLoss(loss_margin)
        else:
            raise ValueError(f"Unknown loss type `{loss}`")
        initializer(self)

    def _to_categorical(self, example, compared_with, sim_bool) -> torch.Tensor:
        # Compute output class
        correct = []
        nb_classes = 2
        for lt, rt, pred in zip(
                example["label"].tolist(),
                compared_with["label"].tolist(),
                sim_bool.tolist()
        ):
            row = [0] * nb_classes  # Categorical expect two classes
            if pred == 1:  # If the pred is 1, the example is predicted to be = to compared_with
                row[rt] += 1
            else:  # Otherwise, it's predicted to != right
                row[int(not rt)] += 1
            correct.append(row)
        return torch.tensor(correct, device=example["label"].device)

    def _pop_metadata(self, inputs: Dict):
        metadata_vector = {}
        if self.left_encoder.use_metadata_vector:
            metadata_vector = {
                cat: inputs.pop(get_metadata_field_name(cat))
                for cat in self.metadata_categories
            }
        return metadata_vector

    def _miner_forward(self, **inputs) -> Dict[str, torch.Tensor]:
        if "label" not in inputs:
            raise ValueError("Miner mode on siamese networks is only available at training and developement time")

        metadata_vector = self._pop_metadata(inputs)
        encoded, additional_output = self.left_encoder(inputs, metadata_vector=metadata_vector)

        mined_pairs: Tuple[torch.Tensor, ...] = self._miner(encoded, inputs["label"])  # Creates pairs
        # Tensor if indices
        anchors_pos, positives, anchors_neg, negatives = mined_pairs
        # There are case where a batch would simply not return anything with specific miners.
        if anchors_pos.shape[0] == 0:
            mined_pairs: Tuple[torch.Tensor, ...] = self._all_miner(encoded, inputs["label"])
            anchors_pos, positives, anchors_neg, negatives = mined_pairs

        loss = self._loss(encoded, inputs["label"], mined_pairs)

        # Tensor of indices
        anchors: torch.Tensor = torch.cat([anchors_pos, anchors_neg], dim=0)
        compars: torch.Tensor = torch.cat([positives, negatives], dim=0)

        # Tensor of content
        cont_anchors = encoded[anchors]
        cont_compars = encoded[compars]

        return {
            "loss": loss,
            **self._shared_compute_metrics(
                left_encoded=cont_anchors, right_encoded=cont_compars,
                left_raw_input=get_tensor_dict_subset(inputs, anchors),
                right_raw_input=get_tensor_dict_subset(inputs, compars),
                equality_label=torch.cat([
                    torch.ones(*anchors_pos.shape, device=encoded.device),
                    torch.zeros(*anchors_neg.shape, device=encoded.device)
                ])
            ),
            **get_tensor_dict_subset(additional_output, anchors)  # Needs to be reshaped to match
        }

    def _shared_compute_metrics(
            self,
            left_encoded, right_encoded,
            left_raw_input, right_raw_input,
            equality_label: Optional[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        dist: torch.Tensor = F.pairwise_distance(left_encoded, right_encoded)
        sim_bool: torch.Tensor = (dist < self.prediction_threshold).long()

        preds: torch.Tensor = self._to_categorical(left_raw_input, right_raw_input, sim_bool)
        self._compute_metrics(
            categorical_predictions=preds,
            categorical_label=left_raw_input["label"].long(),
            similarity_boolean=sim_bool,
            similarity_labels=equality_label
        )

        return {
            "distances": dist,
            "probs": preds
        }

    def _classic_forward(self, **inputs) -> Dict[str, torch.Tensor]:
        left = {key.replace("left_", ""): value for key, value in inputs.items() if key.startswith("left_")}
        right = {key.replace("right_", ""): value for key, value in inputs.items() if key.startswith("right_")}
        label: torch.Tensor = left["label"] == right["label"]

        left_metadata_vector = self._pop_metadata(left)
        right_metadata_vector = self._pop_metadata(right)

        v_l, left_additional_output = self.left_encoder(left, metadata_vector=left_metadata_vector)
        v_r, right_additional_output = self.right_encoder(right, metadata_vector=right_metadata_vector)

        # With pytorch_metric_learning, we need to split positive and negative pairs
        loss = self._loss(v_l, v_r, label)

        return {
            'loss': loss,
            **self._shared_compute_metrics(
                left_encoded=v_l, right_encoded=v_r,
                left_raw_input=left, right_raw_input=right,
                equality_label=label
            ),
            **left_additional_output
        }

    @overrides
    def forward(self,
                **inputs) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------

        Returns
        -------
        An output dictionary consisting of:
        class_probabilities : torch.FloatTensor
            A tensor of shape ``(batch_size, num_classes)`` representing a distribution over the
            label classes for each instance.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """
        if self._reader_mode == "default":
            return self._classic_forward(**inputs)
        else:
            return self._miner_forward(**inputs)

    @overrides
    def _compute_metrics(
            self,
            categorical_predictions: torch.Tensor,
            categorical_label: torch.Tensor,
            similarity_boolean: torch.Tensor,
            similarity_labels: torch.Tensor
    ):

        self._measure(predictions=categorical_predictions, gold_labels=categorical_label)
        for mname, metric in self.metrics.items():
            if mname.startswith("sim"):
                metric(similarity_boolean, similarity_labels)
            else:
                metric(
                    predictions=categorical_predictions,
                    gold_labels=categorical_label
                )

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        beta: Dict[str, List[float]] = self._measure.get_metric(reset)
        out = {
            f"{key}-{self.labels[score_idx]}": score
            for key, scores in beta.items()
            for score_idx, score in enumerate(scores)
        }

        for mname, metric in self.metrics.items():
            out.update(self._get_metrics(mname, metric, reset=reset))

        return out


# ToDo: Check Attention Pooling at
#   https://hanxiao.io/2018/06/24/4-Encoding-Blocks-You-Need-to-Know-Besides-LSTM-RNN-in-Tensorflow/#pooling-block

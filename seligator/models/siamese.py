# https://github.com/PonteIneptique/jdnlp

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy

from typing import Dict, Optional, Tuple, List
from overrides import overrides
from copy import deepcopy
import logging

from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import FeedForward, Seq2VecEncoder, TextFieldEmbedder
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util, Activation
from allennlp.training.metrics import CategoricalAccuracy, FBetaMeasure, BooleanAccuracy

from seligator.modules.loss_functions.constrastiveLoss import ContrastiveLoss

logger = logging.getLogger(__name__)


from seligator.models.base import BaseModel
from seligator.modules.embedders.latinBert import LatinPretrainedTransformer
from seligator.dataset.tokenizer import SubwordTextEncoderTokenizer
from allennlp.modules.seq2vec_encoders import BertPooler, BagOfEmbeddingsEncoder


_Fields = Dict[str, torch.LongTensor]


def distribute(left, label) -> Tuple[_Fields, _Fields, torch.LongTensor]:

    # We split the data in two ?
    batch_size = left["token_subword"].shape[0]

    if not batch_size % 2 == 1:  # If it's odd, we duplicate the last one later
        resized = (batch_size - 1) // 2  # // div with int output
        left, right, odd = torch.split(left["token_subword"], resized)
        left = torch.cat(
            (left, right[random.randint(0, resized)]),  # Pick a random one from right
            dim=0
        )
        right = torch.cat(
            (right, odd),
            dim=0
        )
    else:
        left, right = torch.split(left["token_subword"], batch_size // 2)

    return left, right, label


class SiameseBert(BaseModel):
    def __init__(self,
                 vocab: Vocabulary,
                 input_feature_names: Tuple[str, ...],
                 bert_embedder: LatinPretrainedTransformer,
                 left_encoder: Seq2VecEncoder,
                 right_encoder: Seq2VecEncoder = None,
                 loss_margin: float = 1.0,
                 prediction_threshold: float = 0.6,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 **kwargs):
        super().__init__(vocab, input_feature_names=input_feature_names, regularizer=regularizer)

        self.prediction_threshold = prediction_threshold

        self.bert_embedder: LatinPretrainedTransformer = bert_embedder

        self.left_encoder: Seq2VecEncoder = left_encoder
        self.right_encoder: Seq2VecEncoder = right_encoder or deepcopy(left_encoder)

        if bert_embedder.get_output_dim() != self.left_encoder.get_input_dim():
            raise ConfigurationError("The output dimension of the text_field_embedder must match the "
                                     "input dimension of the encoder. Found {} and {}, "
                                     "respectively.".format(bert_embedder.get_output_dim(),
                                                            self.left_encoder.get_input_dim()))
        self.metrics = {
            "sim-accuracy": BooleanAccuracy(),
            "accuracy": CategoricalAccuracy(),
            #"f1": FBetaMeasure()
        }
        self.loss = ContrastiveLoss(loss_margin)
        initializer(self)

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
        left = {key.replace("left_", ""): value for key, value in inputs.items() if key.startswith("left_")}
        right = {key.replace("right_", ""): value for key, value in inputs.items() if key.startswith("right_")}
        label = left["label"] == right["label"]

        # Deal with Bert (key: token_subword)
        bert_left = left["token_subword"]["token_subword"]
        bert_left_embedded = self.bert_embedder(bert_left["token_ids"], mask=bert_left["mask"])

        bert_right = right["token_subword"]["token_subword"]
        bert_right_embedded = self.bert_embedder(bert_right["token_ids"], mask=bert_right["mask"])

        v_l = self.left_encoder(bert_left_embedded, mask=bert_left["mask"])
        v_r = self.right_encoder(bert_right_embedded, mask=bert_right["mask"])

        loss = self.loss(v_l, v_r, label)
        sim = F.cosine_similarity(v_l, v_r)
        sim_bool = sim > self.prediction_threshold

        output_dict = {
            'loss': loss
        }

        sim_bool = sim_bool.long()
        label = label.long()

        # Compute output class
        correct = []
        for lt, rt, pred in zip(
                left["label"].tolist(),
                right["label"].tolist(),
                sim_bool.tolist()
        ):
            row = [0, 0]  # Categorical expect two classes
            if pred == 1:  # If the pred is 1, left is predicted to be right
                row[rt] += 1
            else:  # Otherwise, it's predicted to not be right
                row[int(not rt)] += 1
            correct.append(row)
        correct = torch.tensor(correct, device=left["label"].device)

        self._measure(predictions=correct, gold_labels=left["label"])
        for mname, metric in self.metrics.items():
            #logging.info(f"Sim {sim_bool}, Label {label}, Left {left['label']}, Right {right['label']}")
            #logging.info(f"Label {label}")
            if mname.startswith("sim"):
                metric(sim_bool, label)
            else:
                metric(
                    predictions=correct,#, dtype=torch.LongTensor),
                    gold_labels=left["label"].long()
                )

        return output_dict

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

    #def get_metrics(self, reset: bool = False) -> Dict[str, float]:
    #    return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}


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


def build_model(
        vocab: Vocabulary,
        *,
        use_only: Tuple[str, ...] = ("token_subword", ),
        bert_dir: str = None,
        bert_tokenizer: Optional[SubwordTextEncoderTokenizer] = None,
        **kwargs
) -> Model:

    if use_only != ("token_subword", ):
        raise Exception("Currently, it's impossible to use something else than token subword")

    bert = LatinPretrainedTransformer(bert_dir, tokenizer=bert_tokenizer, train_parameters=False)
    # bert_pooler = BertPooler(bert_dir)
    bert_pooler = SumAndLinear(bert.get_output_dim())

    return SiameseBert(
        vocab=vocab,
        bert_embedder=bert,
        left_encoder=bert_pooler,
        input_feature_names=use_only
    )


if __name__ == "__main__":
    from seligator.training.trainer import run_training_loop
    import logging
    logging.getLogger().setLevel(logging.INFO)
    model, dataset_reader = run_training_loop(
        build_model=build_model,
        cuda_device=0,
        batch_size=6,
        bert_dir="bert/latin_bert",
        use_only=("token_subword", ),
        siamese=True
    )

# https://github.com/PonteIneptique/jdnlp

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy

from typing import Dict, Optional, Tuple, List, Callable, Union
from overrides import overrides
from copy import deepcopy
import logging

from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Seq2VecEncoder, TextFieldEmbedder
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder
from allennlp.nn import InitializerApplicator, RegularizerApplicator, util
from allennlp.training.metrics import CategoricalAccuracy, BooleanAccuracy

from seligator.models.base import BaseModel
from seligator.modules.embedders.latinBert import LatinPretrainedTransformer
from seligator.dataset.tokenizer import SubwordTextEncoderTokenizer
from seligator.modules.loss_functions.constrastiveLoss import ContrastiveLoss

logger = logging.getLogger(__name__)


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


Left_Or_LeftAndRight_Encoder = Optional[Union[Seq2VecEncoder, Tuple[Seq2VecEncoder, Seq2VecEncoder]]]


class SiameseClassifier(BaseModel):
    BERT_COMPATIBLE: bool = True
    IS_SIAMESE: bool = True

    def __init__(self,
                 vocab: Vocabulary,
                 input_features: Tuple[str, ...],

                 # Normal input
                 use_features: bool = True,
                 features_embedder: Optional[TextFieldEmbedder] = None,
                 features_encoder: Left_Or_LeftAndRight_Encoder = None,

                 # Bert Input
                 use_bert: bool = False,
                 bert_embedder: Optional[LatinPretrainedTransformer] = None,
                 bert_pooler: Left_Or_LeftAndRight_Encoder = None,

                 mixer: str = "concat",

                 emb_dropout: float = 0.3,

                 loss_margin: float = 1.0,
                 prediction_threshold: float = 0.6,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 **kwargs):
        super().__init__(vocab, input_features=input_features)

        if emb_dropout:
            self._emb_dropout = torch.nn.Dropout(emb_dropout)
        else:
            self._emb_dropout = None

        self.prediction_threshold = prediction_threshold
        self.regularizer: Optional[RegularizerApplicator] = RegularizerApplicator

        self.use_features: bool = use_features
        self.features_embedder = features_embedder
        self.left_features_encoder: Optional[Seq2VecEncoder] = None
        self.right_features_encoder: Optional[Seq2VecEncoder] = None

        if use_features:
            logger.info("Current model uses features")
            if isinstance(features_encoder, Seq2VecEncoder):
                self.left_features_encoder = features_encoder
                self.right_features_encoder = deepcopy(features_encoder)
            elif isinstance(features_encoder, tuple) and len(features_encoder) == 2:
                self.left_features_encoder, self.right_features_encoder = features_encoder
            elif use_features:
                raise ConfigurationError("Param `features_encoder` should be a single Seq2Vec encoder or "
                                         "a tuple of two. Invalid value passed")

            # Check everything is okay
            if self.features_embedder.get_output_dim() != self.left_features_encoder.get_input_dim():
                raise ConfigurationError("The output dimension of the text_field_embedder must match the "
                                         "input dimension of the encoder. Found {} and {}, "
                                         "respectively.".format(features_embedder.get_output_dim(),
                                                                self.left_features_encoder.get_input_dim()))

        self.use_bert: bool = use_bert
        self.bert_embedder: LatinPretrainedTransformer = bert_embedder
        self.left_bert_pooler: Optional[Seq2VecEncoder] = None
        self.right_bert_pooler: Optional[Seq2VecEncoder] = None

        if use_bert:
            logger.info("Current model uses Bert")
            if isinstance(bert_pooler, Seq2VecEncoder):
                self.left_bert_pooler = bert_pooler
                self.right_bert_pooler = deepcopy(bert_pooler)
            elif isinstance(bert_pooler, tuple) and len(bert_pooler) == 2:
                self.left_bert_pooler, self.right_bert_pooler = bert_pooler
            elif use_features:
                raise ConfigurationError("Param `bert_pooler` should be a single Seq2Vec encoder or "
                                         "a tuple of two. Invalid value passed")

            # Checking everything is okay
            if self.bert_embedder.get_output_dim() != self.left_bert_pooler.get_input_dim():
                raise ConfigurationError("The output dimension of the text_field_embedder must match the "
                                         "input dimension of the encoder. Found {} and {}, "
                                         "respectively.".format(bert_embedder.get_output_dim(),
                                                                self.left_encoder.get_input_dim()))

        out_dim = 0
        if use_features and use_bert:
            if mixer == "concat":
                self.mixer = lambda inp1, inp2: torch.cat([inp1, inp2], -1)
                out_dim = self.left_features_encoder.get_output_dim() + self.left_bert_pooler.get_output_dim()
            elif mixer.startswith("Linear:"):
                out_dim = int(mixer.split(":")[-1])
                self._mixer = torch.nn.Linear(
                    self.left_features_encoder.get_output_dim() + self.left_bert_pooler.get_output_dim(),
                    out_dim
                )
                self.mixer = lambda inp1, inp2: self._mixer(torch.cat([inp1, inp2], -1))
        elif use_features:
            out_dim = self.left_features_encoder.get_output_dim()
        elif use_bert:
            out_dim = self.left_bert_pooler.get_output_dim()
        else:
            raise ValueError("Neither Bert or Features are used. Make sure to set either `use_features` "
                             "or `use_bert` to True.")

        self.encoder_out_dim: int = out_dim

        self.metrics = {
            "sim-accuracy": BooleanAccuracy(),
            "accuracy": CategoricalAccuracy(),
            #"f1": FBetaMeasure()
        }
        self.loss = ContrastiveLoss(loss_margin)
        initializer(self)

    @classmethod
    def build_model(
            cls,
            vocabulary: Vocabulary,
            emb_dims: Dict[str, int] = None,
            input_features: Tuple[str, ...] = ("token",),
            features_encoder: Callable[[int], Seq2VecEncoder] = BagOfEmbeddingsEncoder,
            char_encoders: Dict[str, Seq2VecEncoder] = None,
            **kwargs
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
            features_encoder=features_encoder(emb.get_output_dim()),
            **kwargs
        )

    def _forward_bert(self, token, encoder: Seq2VecEncoder):
        embedded = self.bert_embedder(token["token_ids"], mask=token["mask"])
        return encoder(embedded, mask=token["mask"])

    def _forward_features(self, token, encoder: Seq2VecEncoder):
        token = self._rebuild_input(token)

        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.features_embedder(token)
        if self._emb_dropout is not None:
            # Shape: (batch_size, num_tokens, embedding_dim)
            embedded_text = self._emb_dropout(embedded_text)
        # Shape: (batch_size, num_tokens)
        mask = util.get_text_field_mask(token)
        # Shape: (batch_size, encoding_dim)
        return encoder(embedded_text, mask)

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
        if self.use_bert and self.use_features:
            bert_left = self._forward_bert(left["token_subword"]["token_subword"], encoder=self.left_bert_pooler)
            bert_right = self._forward_bert(right["token_subword"]["token_subword"], encoder=self.right_bert_pooler)
            features_left = self._forward_features(left, encoder=self.left_features_encoder)
            features_right = self._forward_features(right, encoder=self.right_features_encoder)
            v_l = self.mixer(bert_left, features_left)
            v_r = self.mixer(bert_right, features_right)
        elif self.use_bert:
            v_l = self._forward_bert(left["token_subword"]["token_subword"], encoder=self.left_bert_pooler)
            v_r = self._forward_bert(right["token_subword"]["token_subword"], encoder=self.right_bert_pooler)
        elif self.use_features:
            v_l = self._forward_features(left, encoder=self.left_features_encoder)
            v_r = self._forward_features(right, encoder=self.right_features_encoder)
        else:
            raise ValueError("No features or bert used.")

        loss = self.loss(v_l, v_r, label)
        sim = F.cosine_similarity(v_l, v_r)
        sim_bool = sim > self.prediction_threshold

        output_dict = {'loss': loss}

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
            if mname.startswith("sim"):
                metric(sim_bool, label)
            else:
                metric(
                    predictions=correct,
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


if __name__ == "__main__":
    import logging
    logging.getLogger().setLevel(logging.INFO)
    from allennlp.modules.seq2vec_encoders import LstmSeq2VecEncoder
    from seligator.training.trainer import generate_all_data, train_model
    from allennlp.modules.seq2vec_encoders import LstmSeq2VecEncoder, BertPooler
    from seligator.common.constants import EMBEDDING_DIMENSIONS

    # For test, just change the input feature here
    # INPUT_FEATURES = ("token", "lemma", "token_char",)
    INPUT_FEATURES = ("token_subword", )
    USE_BERT = True
    BERT_DIR = "./bert/latin_bert"

    train, dev, vocab, reader = generate_all_data(
        input_features=INPUT_FEATURES, bert_dir=BERT_DIR,
        is_siamese=True
    )
    bert, bert_pooler = None, None
    if USE_BERT:
        bert = LatinPretrainedTransformer(
            BERT_DIR,
            tokenizer=reader.token_indexers["token_subword"].tokenizer,
            train_parameters=False
        )
        bert_pooler = BertPooler(BERT_DIR)

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

    model = SiameseClassifier.build_model(
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
        char_encoders=embedding_encoders,
        use_features=bool(len([x for x in INPUT_FEATURES if x not in {"token_subword"}])),
        use_bert=USE_BERT,
        bert_embedder=bert,
        bert_pooler=bert_pooler
    )

    model.cuda()

    train_model(
        model=model,
        train_loader=train,
        dev_loader=dev,
        cuda_device=0
    )

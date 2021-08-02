# https://github.com/PonteIneptique/jdnlp

import torch
import torch.nn.functional as F
import random

from typing import Dict, Optional, Tuple, List, Union
from overrides import overrides
import logging

from allennlp.data import Vocabulary
from allennlp.modules import Seq2VecEncoder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import CategoricalAccuracy, BooleanAccuracy

from seligator.models.base import BaseModel
from seligator.modules.mixed_encoders import MixedEmbeddingEncoder
from seligator.modules.loss_functions.constrastiveLoss import ContrastiveLoss
from seligator.common.params import get_metadata_field_name, BasisVectorConfiguration

logger = logging.getLogger(__name__)


_Fields = Dict[str, torch.LongTensor]


class SiameseClassifier(BaseModel):
    BERT_COMPATIBLE: bool = True
    IS_SIAMESE: bool = True
    INSTANCE_TYPE: str = "siamese"

    def __init__(self,
                 vocab: Vocabulary,
                 input_features: Tuple[str, ...],
                 mixed_encoder: Union[MixedEmbeddingEncoder, Tuple[MixedEmbeddingEncoder, MixedEmbeddingEncoder]],

                 loss_margin: float = 0.3,
                 prediction_threshold: float = 0.7,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 basis_vector_configuration: BasisVectorConfiguration = None,
                 loss: str = "contrastive",
                 **kwargs):
        super().__init__(vocab, input_features=input_features)

        self.prediction_threshold = prediction_threshold
        self.regularizer: Optional[RegularizerApplicator] = regularizer

        self.left_encoder: Optional[Seq2VecEncoder] = None
        self.right_encoder: Optional[Seq2VecEncoder] = None

        self.metadata_categories: Tuple[str, ...] = basis_vector_configuration.categories_tuple

        if isinstance(mixed_encoder, MixedEmbeddingEncoder):
            self.left_encoder = mixed_encoder
            self.right_encoder = mixed_encoder.copy_for_siamese()
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
        #example = example["label"].tolist()
        #compared_with = compared_with["label"]
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

    def _to_categorical_probs(self, example, compared_with, sim) -> torch.Tensor:
        # Compute output class
        correct = []
        #example = example["label"].tolist()
        #compared_with = compared_with["label"]
        nb_classes = 2
        for lt, rt, pred in zip(
                example["label"].tolist(),
                compared_with["label"].tolist(),
                sim.tolist()
        ):
            row = [.0, .0]
            row[int(not rt)] = abs(1 - pred)
            row[rt] = pred
            correct.append(row)
        return torch.tensor(correct, device=example["label"].device)

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

        metadata_vector = {}
        if self.left_encoder.use_metadata_vector:
            metadata_vector = {
                cat: inputs.pop("left_"+get_metadata_field_name(cat))
                for cat in self.metadata_categories
            }

        v_l, left_additional_output = self.left_encoder(left, metadata_vector=metadata_vector)
        v_r, right_additional_output = self.right_encoder(right, metadata_vector=metadata_vector)

        loss = self._loss(v_l, v_r, label)
        sim = F.cosine_similarity(v_l, v_r)
        sim_bool = sim > self.prediction_threshold

        output_dict = {
            'loss': loss,
            "sim": sim,
            "probs": self._to_categorical_probs(left, right, sim),
            **left_additional_output
        }

        sim_bool = sim_bool.long()
        label = label.long()

        self._compute_metrics(
            categorical_predictions=self._to_categorical(left, right, sim_bool),
            categorical_label=left["label"].long(),
            similarity_boolean=sim,
            similarity_labels=label
        )

        return output_dict

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


# Check Attention Pooling at https://hanxiao.io/2018/06/24/4-Encoding-Blocks-You-Need-to-Know-Besides-LSTM-RNN-in-Tensorflow/#pooling-block

if __name__ == "__main__":
    import logging
    logging.getLogger().setLevel(logging.INFO)
    from allennlp.modules.token_embedders.pretrained_transformer_embedder import PretrainedTransformerEmbedder
    from allennlp.modules.seq2vec_encoders import LstmSeq2VecEncoder, CnnEncoder
    from seligator.training.trainer import generate_all_data, train_model
    from allennlp.modules.seq2vec_encoders import LstmSeq2VecEncoder, BertPooler, CnnHighwayEncoder
    from seligator.common.constants import EMBEDDING_DIMENSIONS
    from seligator.common.bert_utils import what_type_of_bert
    from seligator.modules.seq2vec.bert_poolers import PoolerHighway

    # For test, just change the input feature here
    #INPUT_FEATURES = ("token", "lemma", "token_char",)
    INPUT_FEATURES = ("token_subword", )
    USE_BERT = "token_subword" in INPUT_FEATURES
    BERT_DIR = "./bert/latin_bert"
    HUGGIN = False

    if USE_BERT:
        if HUGGIN:
            get_me_bert = what_type_of_bert("ponteineptique/latin-classical-small", trainable=False, hugginface=True)
            print("Using hugginface !")
        else:
            get_me_bert = what_type_of_bert(BERT_DIR, trainable=False, hugginface=False)
    else:
        get_me_bert = what_type_of_bert()

    train, dev, vocab, reader = generate_all_data(
        input_features=INPUT_FEATURES, get_me_bert=get_me_bert,
        instance_type="siamese", batches_per_epoch=20, batch_size=64
    )
    bert, bert_pooler = None, None
    if USE_BERT:
        bert = get_me_bert.embedder
        #bert = PretrainedTransformerEmbedder("ponteineptique/latin-classical-small")
        #bert_pooler = BertPooler(BERT_DIR)
        #bert_pooler = PoolerHighway(CnnEncoder(
        #    embedding_dim=bert.get_output_dim(),
        #    num_filters=374,
        #), 128)
        bert_pooler = PoolerHighway(BertPooler(BERT_DIR), 128)

        # Max Pooler
        #bert_pooler = CnnEncoder(
        #    embedding_dim=bert.get_output_dim(),
        #    num_filters=128,
        #)
        #bert_pooler = CnnHighwayEncoder(
        #    embedding_dim=bert.get_output_dim(),
        #    num_filters=128,
        #)

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

    model = SiameseClassifier(
        vocab=vocab,
        input_features=INPUT_FEATURES,
        mixed_encoder=MixedEmbeddingEncoder.build(
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

            use_bert=USE_BERT,
            bert_embedder=bert,
            bert_pooler=bert_pooler
        )
    )

    model.cuda()
    print(model)
    train_model(
        model=model,
        train_loader=train,
        dev_loader=dev,
        cuda_device=0,
        num_epochs=200,
        lr=3e-5
    )

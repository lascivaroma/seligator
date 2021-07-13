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

logger = logging.getLogger(__name__)


_Fields = Dict[str, torch.LongTensor]


class SiameseClassifier(BaseModel):
    BERT_COMPATIBLE: bool = True
    IS_SIAMESE: bool = True

    def __init__(self,
                 vocab: Vocabulary,
                 input_features: Tuple[str, ...],
                 mixed_encoder: Union[MixedEmbeddingEncoder, Tuple[MixedEmbeddingEncoder, MixedEmbeddingEncoder]],

                 loss_margin: float = 1.0,
                 prediction_threshold: float = 0.6,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 loss: str = "contrastive",
                 **kwargs):
        super().__init__(vocab, input_features=input_features)

        self.prediction_threshold = prediction_threshold
        self.regularizer: Optional[RegularizerApplicator] = regularizer

        self.left_encoder: Optional[Seq2VecEncoder] = None
        self.right_encoder: Optional[Seq2VecEncoder] = None

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

        v_l, left_additional_output = self.left_encoder(left)
        v_r, right_additional_output = self.right_encoder(right)

        loss = self._loss(v_l, v_r, label)
        sim = F.cosine_similarity(v_l, v_r)
        sim_bool = sim > self.prediction_threshold

        output_dict = {'loss': loss, "sim": sim}

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

    # For test, just change the input feature here
    INPUT_FEATURES = ("token", "lemma", "token_char",)
    # INPUT_FEATURES = ("token_subword", )
    USE_BERT = False
    BERT_DIR = "./bert/latin_bert"
    HUGGIN = True

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
        is_siamese=True
    )
    bert, bert_pooler = None, None
    if USE_BERT:
        bert = get_me_bert.embedder
        #bert = PretrainedTransformerEmbedder("ponteineptique/latin-classical-small")

        # bert_pooler = BertPooler(BERT_DIR, dropout=0.3)
        # Max Pooler
        bert_pooler = CnnEncoder(
            embedding_dim=bert.get_output_dim(),
            num_filters=128,
        )
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
        cuda_device=0
    )

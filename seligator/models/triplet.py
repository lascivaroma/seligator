# https://github.com/PonteIneptique/jdnlp

import torch.nn.functional as F
from torch import Tensor, cat

from typing import Dict, Optional, Tuple, List, Callable, Union
import logging
from overrides import overrides
from pytorch_metric_learning import losses, miners

from seligator.models.siamese import SiameseClassifier
from seligator.modules.loss_functions.tripletLoss import TripletLoss

logger = logging.getLogger(__name__)


class TripletClassifier(SiameseClassifier):
    BERT_COMPATIBLE: bool = True
    IS_SIAMESE: bool = True
    TRIPLET: bool = True

    def __init__(self, *args, **kwargs):
        super(TripletClassifier, self).__init__(*args, **kwargs)
        #self._loss = TripletLoss(kwargs.get("loss_margin", 1.0))
        self._loss = losses.TripletMarginLoss(
            margin=kwargs.get("loss_margin", 1.0),
            triplets_per_anchor="all",
        )

    @overrides
    def forward(
            self,
            **inputs
    ) -> Dict[str, Tensor]:

        example = self.filter_input_dict(inputs, "left_")
        # ToDo: Right now, positiv and negative are single examples. Should we check that its all the same
        #       and reduce the size of the matrix for computation ?
        positiv = self.filter_input_dict(inputs, "positive_")
        negativ = self.filter_input_dict(inputs, "negative_")

        label = None
        if example.get("label", None) is not None:
            label = (example["label"] == positiv["label"]).long()

        # Need to take care of label ?
        exm, example_additional_output = self.left_encoder(example)
        pos, positiv_additional_output = self.right_encoder(positiv)
        neg, negativ_additional_output = self.right_encoder(negativ)

        out = {
            "pos_similarity": F.cosine_similarity(exm, pos),
        }
        if negativ:
            out["neg_similarity"] = F.cosine_similarity(exm, neg)

        positiv_sim_bool = out["pos_similarity"] > self.prediction_threshold

        if label is not None:
            sim_bool = positiv_sim_bool.long()
            if isinstance(self._loss, losses.BaseMetricLossFunction):
                # print(exm["label"], pos["label"])
                encoded = cat([exm, pos[:1, :], neg[:1, :]], dim=0) # We only keep the first, hack for ToDo
                out["loss"] = self._loss(
                    encoded,
                    cat([example["label"], positiv["label"][:1], negativ["label"][:1]], dim=0)
                ) or (encoded * 0).sum()
            else:
                out["loss"] = self._loss(exm, pos, neg)

            self._compute_metrics(
                categorical_predictions=self._to_categorical(example, positiv, sim_bool),
                categorical_label=example["label"].long(),
                similarity_boolean=sim_bool,
                similarity_labels=label
            )

        return out


# Check Attention Pooling at
#    https://hanxiao.io/2018/06/24/4-Encoding-Blocks-You-Need-to-Know-Besides-LSTM-RNN-in-Tensorflow/#pooling-block

if __name__ == "__main__":
    import logging
    logging.getLogger().setLevel(logging.INFO)

    from allennlp.modules.seq2vec_encoders import LstmSeq2VecEncoder, CnnEncoder
    from seligator.training.trainer import generate_all_data, train_model
    from allennlp.modules.seq2vec_encoders import LstmSeq2VecEncoder
    from seligator.common.constants import EMBEDDING_DIMENSIONS
    from seligator.common.bert_utils import what_type_of_bert
    from seligator.modules.mixed_encoders import MixedEmbeddingEncoder

    # For test, just change the input feature here
    #INPUT_FEATURES = ("token", "lemma", "token_char",)
    INPUT_FEATURES = ("token_subword", )
    USE_BERT = True
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
        instance_type="triplet"
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

    model = TripletClassifier(
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

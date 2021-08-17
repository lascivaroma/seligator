# https://github.com/PonteIneptique/jdnlp

import torch.nn.functional as F
from torch import Tensor, cat, tensor
from torch.nn import TripletMarginLoss as TorchTripletMarginLoss
from pytorch_metric_learning.losses import TripletMarginLoss as PTLTripletMarginLoss
from pytorch_metric_learning.miners import BatchEasyHardMiner, BatchHardMiner

from typing import Dict, Optional, Tuple, List, Callable, Union
import logging
from overrides import overrides

from seligator.models.siamese import SiameseClassifier, get_tensor_dict_subset

logger = logging.getLogger(__name__)


class TripletClassifier(SiameseClassifier):
    BERT_COMPATIBLE: bool = True
    IS_SIAMESE: bool = True
    TRIPLET: bool = True
    INSTANCE_TYPE: str = "triplet"

    def __init__(self, *args, **kwargs):
        super(TripletClassifier, self).__init__(*args, **kwargs)

        #self._loss = TripletLoss(kwargs.get("loss_margin", 1.0))
        if self._reader_mode == "default":
            self._loss = TorchTripletMarginLoss(
                margin=kwargs.get("loss_margin", 1.0),
                p=kwargs.get("loss_p", 2),
                #triplets_per_anchor="all",
            )
        elif self._reader_mode == "miner":
            self._loss = PTLTripletMarginLoss(margin=kwargs.get("loss_margin", 1.0))
            self._miner = BatchHardMiner()
            self._all_miner = BatchEasyHardMiner(
                pos_strategy=BatchEasyHardMiner.ALL,
                neg_strategy=BatchEasyHardMiner.ALL
            )

    def _to_categorical_probs(self, negative: Dict, positive: Dict, neg_dist: Tensor, pos_dist: Tensor) -> Tensor:
        # Compute output class
        correct = []
        for neg, dist_neg, pos, dist_pos in zip(
            negative["label"],
            neg_dist.tolist(),
            positive["label"],
            pos_dist.tolist()
        ):
            dist_total = abs(dist_neg) + abs(dist_pos)
            row = [.0, .0]
            # To make it more like "Fake probabilities", probability to be of positive is
            # is 1-dist/sum(all dist). We remove it from one because the lowest distance
            # is supposed to be the closest result (hence lower distance = higher probability)
            row[neg] = 1 - abs(dist_neg) / dist_total
            row[pos] = 1 - abs(dist_pos) / dist_total
            correct.append(row)
        return tensor(correct, device=neg_dist.device)

    def _miner_forward(self, **inputs) -> Dict[str, Tensor]:
        encoded, additional_output = self.left_encoder(inputs)
        # MinedTriplets = 3 tensor of indexes
        if self.training:
            anc, pos, neg = self._miner(encoded, inputs["label"])
            if anc.shape[0] == 0:
                anc, pos, _, neg = self._all_miner(encoded, inputs["label"])
        else:
            anc, pos, anc2, neg = self._all_miner(encoded, inputs["label"]) # This if fucked up
        loss = self._loss(encoded, inputs["label"], (anc, pos, neg))

        # Get content
        anc_content = encoded[anc]
        pos_content = encoded[pos]
        neg_content = encoded[neg]

        anc_raw = get_tensor_dict_subset(inputs, anc)
        pos_raw = get_tensor_dict_subset(inputs, pos)
        neg_raw = get_tensor_dict_subset(inputs, neg)

        out = {
            "loss": loss,
            **self._shared_output_with_probs(
                anc=anc_content, pos=pos_content, neg=neg_content,
                positive_raw=pos_raw, negative_raw=neg_raw
            )
        }

        self._shared_compute_metrics(out=out, anc_raw_input=anc_raw, pos_raw_input=pos_raw)
        return out

    def _shared_output_with_probs(
            self,
            anc, pos, neg,
            negative_raw, positive_raw
    ) -> Dict[str, Tensor]:
        out = {
            "pos_distance": F.pairwise_distance(anc, pos),  # Euclidian
            "neg_distance": F.pairwise_distance(anc, neg),  # Euclidian
            "doc-vectors": anc.tolist()
        }
        out["pos-closer"] = out["pos_distance"] < out["neg_distance"]
        out["probs"] = self._to_categorical_probs(
            negative=negative_raw, positive=positive_raw,
            pos_dist=out["pos_distance"],
            neg_dist=out["neg_distance"]
        )

        return out

    @overrides
    def _shared_compute_metrics(
        self,
        out,
        anc_raw_input,
        pos_raw_input
    ) -> None:

        self._compute_metrics(
            categorical_predictions=out["probs"],
            categorical_label=anc_raw_input["label"].long(),
            similarity_boolean=out["pos-closer"],
            similarity_labels=pos_raw_input["label"].long()
        )

    def _classic_forward(
        self,
        **inputs
    ) -> Dict[str, Tensor]:

        example = self.filter_input_dict(inputs, "left_")
        # ToDo: Right now, positiv and negative are single examples. Should we check that its all the same
        #       and reduce the size of the matrix for computation ?
        positiv = self.filter_input_dict(inputs, "positive_")
        negativ = self.filter_input_dict(inputs, "negative_")

        # Need to take care of label ?
        exm, example_additional_output = self.left_encoder(example)
        pos, positiv_additional_output = self.right_encoder(positiv)
        neg, negativ_additional_output = self.right_encoder(negativ)

        out = self._shared_output_with_probs(
            anc=exm, pos=pos, neg=neg,
            negative_raw=negativ, positive_raw=positiv
        )
        out["loss"] = self._loss(exm, pos, neg)

        self._shared_compute_metrics(out=out, anc_raw_input=example, pos_raw_input=positiv)

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

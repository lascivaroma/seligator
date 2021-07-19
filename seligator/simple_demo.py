from typing import Tuple

from allennlp.data import DataLoader, DatasetReader
from allennlp.modules.seq2vec_encoders import LstmSeq2VecEncoder, BertPooler

from seligator.models.classifier import FeatureEmbeddingClassifier
from seligator.modules.mixed_encoders import MixedEmbeddingEncoder
from seligator.training.trainer import generate_all_data, train_model
from seligator.common.constants import EMBEDDING_DIMENSIONS
from seligator.common.bert_utils import what_type_of_bert
from seligator.modules.seq2vec.han import HierarchicalAttentionalEncoder
from seligator.modules.seq2vec.bert_poolers import PoolerHighway, SumAndLinear

import logging

logging.getLogger().setLevel(logging.INFO)


def prepare_model(
    input_features: Tuple[str, ...] = ("token_subword",),
    bert_dir: str = "./bert/latin_bert",
    use_han: bool = False,
    model_class = FeatureEmbeddingClassifier,
    use_bert_higway: bool = False
) -> Tuple[FeatureEmbeddingClassifier, DatasetReader, DataLoader, DataLoader]:

    use_bert = "token_subword" in input_features
    # For test, just change the input feature here
    # INPUT_FEATURES = ("token", "lemma", "token_char")  # , "token_subword")

    if use_han:
        def features_encoder(input_dim):
            return HierarchicalAttentionalEncoder(
                input_dim=input_dim,
                hidden_size=128
            )
    else:
        def features_encoder(input_dim):
            return LstmSeq2VecEncoder(
                input_size=input_dim,
                hidden_size=128,
                dropout=0.3,
                bidirectional=True,
                num_layers=2
            )

    if use_bert:
        get_me_bert = what_type_of_bert(bert_dir, trainable=False, hugginface=False)
    else:
        get_me_bert = what_type_of_bert()

    train, dev, vocab, reader = generate_all_data(
        input_features=input_features,
        get_me_bert=get_me_bert,
        instance_type=model_class.INSTANCE_TYPE
    )

    bert, bert_pooler = None, None
    if use_bert:
        bert = get_me_bert.embedder
        if use_bert_higway:
            bert_pooler = PoolerHighway(BertPooler(bert_dir), 128)
        else:
            bert_pooler = BertPooler(bert_dir)

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

    model = model_class(
        vocab=vocab,
        input_features=input_features,
        mixed_encoder=MixedEmbeddingEncoder.build(
            vocabulary=vocab,
            emb_dims=MixedEmbeddingEncoder.merge_default_embeddings({"token": 100}),
            pretrained_embeddings={"token": "~/Downloads/latin.embeddings"},
            trainable_embeddings={"token": False},
            input_features=input_features,
            features_encoder=features_encoder,
            char_encoders=embedding_encoders,

            use_bert=use_bert,
            bert_embedder=bert,
            bert_pooler=bert_pooler
        )
    )
    return model, reader, train, dev


def train_and_get(model, train, dev, lr:float = 1e-4) -> FeatureEmbeddingClassifier:
    model.cuda()

    train_model(
        model=model,
        train_loader=train,
        dev_loader=dev,
        cuda_device=0,
        lr=lr
    )
    return model

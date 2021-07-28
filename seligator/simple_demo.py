from typing import Tuple, Optional, Dict, Any

from allennlp.data import DataLoader, DatasetReader
from allennlp.modules.seq2vec_encoders import LstmSeq2VecEncoder, BertPooler

from seligator.models.classifier import FeatureEmbeddingClassifier
from seligator.modules.mixed_encoders import MixedEmbeddingEncoder
from seligator.training.trainer import generate_all_data, train_model
from seligator.common.constants import EMBEDDING_DIMENSIONS
from seligator.common.params import Seq2VecEncoderType, BasisVectorConfiguration, MetadataEncoding
from seligator.common.bert_utils import what_type_of_bert

from seligator.modules.basis_customizer import MetadataEnrichedAttentionalLSTM
from seligator.modules.seq2vec.han import HierarchicalAttentionalEncoder
from seligator.modules.seq2vec.bert_poolers import PoolerHighway, SumAndLinear

import logging

logging.getLogger().setLevel(logging.INFO)


def prepare_model(
    input_features: Tuple[str, ...] = ("token_subword",),
    bert_dir: str = "./bert/latin_bert",
    seq2vec_encoder_type: Seq2VecEncoderType = Seq2VecEncoderType.LSTM,
    model_class = FeatureEmbeddingClassifier,
    additional_model_kwargs: Dict[str, Any] = None,
    batches_per_epoch: Optional[int] = None,
    reader_kwargs: Dict[str, Any] = None,
    encoder_hidden_size: int = 64,
    agglomerate_msd: bool = False,
    use_bert_higway: bool = False,
    model_embedding_kwargs: Optional[Dict[str, Any]] = None,
    basis_vector_configuration: BasisVectorConfiguration = None,
) -> Tuple[FeatureEmbeddingClassifier, DatasetReader, DataLoader, DataLoader]:

    use_bert = "token_subword" in input_features
    # For test, just change the input feature here
    # INPUT_FEATURES = ("token", "lemma", "token_char")  # , "token_subword")

    if seq2vec_encoder_type == Seq2VecEncoderType.HAN:
        def features_encoder(input_dim):
            return HierarchicalAttentionalEncoder(
                input_dim=input_dim,
                hidden_size=encoder_hidden_size
            )
    elif seq2vec_encoder_type in {
        Seq2VecEncoderType.AttentionPooling,
        Seq2VecEncoderType.MetadataAttentionPooling,
        Seq2VecEncoderType.MetadataLSTM
    }:
        def features_encoder(input_dim):
            return MetadataEnrichedAttentionalLSTM(
                input_dim=input_dim,
                hidden_size=encoder_hidden_size,
                use_metadata_attention=seq2vec_encoder_type == Seq2VecEncoderType.MetadataAttentionPooling,
                use_metadata_lstm=seq2vec_encoder_type == Seq2VecEncoderType.MetadataLSTM,
                basis_vector_configuration=basis_vector_configuration
            )
    else:
        def features_encoder(input_dim):
            return LstmSeq2VecEncoder(
                input_size=input_dim,
                hidden_size=encoder_hidden_size,
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
        instance_type=model_class.INSTANCE_TYPE,
        batches_per_epoch=batches_per_epoch,
        **{**(reader_kwargs or {}), "agglomerate_msd": agglomerate_msd}
    )

    if basis_vector_configuration:
        logging.info("Fitting the BasisVectorConfiguration")
        basis_vector_configuration.set_metadata_categories_dims(vocab)

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
        if cat in input_features
    }

    if input_features != reader.categories:  # eg. Might have been changed by agglomeration
        input_features = reader.categories

    model = model_class(
        vocab=vocab,
        input_features=input_features,
        mixed_encoder=MixedEmbeddingEncoder.build(
            vocabulary=vocab,
            emb_dims=MixedEmbeddingEncoder.merge_default_embeddings(
                model_embedding_kwargs.pop("pretrained_emb_dims")
            ),
            input_features=input_features,
            features_encoder=features_encoder,
            char_encoders=embedding_encoders,
            use_bert=use_bert,
            bert_embedder=bert,
            bert_pooler=bert_pooler,
            model_embedding_kwargs=model_embedding_kwargs
        ),
        basis_vector_configuration=basis_vector_configuration,
        **(additional_model_kwargs or {})
    )
    return model, reader, train, dev


def train_and_get(model, train, dev, lr: float = 1e-4, use_cpu: bool = False,  **train_kwargs) -> FeatureEmbeddingClassifier:
    if not use_cpu:
        model.cuda()

    train_model(
        model=model,
        train_loader=train,
        dev_loader=dev,
        cuda_device=0 if not use_cpu else -1,
        lr=lr,
        **train_kwargs
    )
    return model

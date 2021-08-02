from typing import Tuple, Optional, Dict, Any, Callable

from torch import nn

from allennlp.data import DataLoader, DatasetReader
from allennlp.modules.seq2vec_encoders import LstmSeq2VecEncoder, BertPooler

from seligator.common.load_save import load, merge
from seligator.common.constants import EMBEDDING_DIMENSIONS
from seligator.common.params import Seq2VecEncoderType, BasisVectorConfiguration, MetadataEncoding
from seligator.common.bert_utils import what_type_of_bert, GetMeBert

from seligator.models.classifier import FeatureEmbeddingClassifier
from seligator.modules.mixed_encoders import MixedEmbeddingEncoder
from seligator.training.trainer import generate_all_data, train_model
from seligator.modules.seq2vec import \
    HierarchicalAttentionalEncoder, PoolerHighway, SumAndLinear, MetadataEnrichedAttentionalLSTM

import logging

logging.getLogger().setLevel(logging.INFO)


class Seligator:
    def __init__(self, model):
        self.model: FeatureEmbeddingClassifier = model
        self.vocabulary = self.model.vocab

    @staticmethod
    def get_defaults():
        return load("best.json")


    @staticmethod
    def _get_seq2vec_encoder(
            seq2vec_encoder_type: Seq2VecEncoderType,
            encoder_hidden_size: int,
            basis_vector_configuration: BasisVectorConfiguration
    ) -> Callable[[int], nn.Module]:

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
        return features_encoder

    @staticmethod
    def _get_me_bert(use: bool, highway: bool, bert_dir:str = None) ->Tuple[GetMeBert, nn.Module, nn.Module]:
        bert, bert_pooler = None, None
        if use:
            get_me_bert = what_type_of_bert(bert_dir, trainable=False, hugginface=False)
            if use:
                bert = get_me_bert.embedder
                if highway:
                    bert_pooler = PoolerHighway(BertPooler(bert_dir), 256)
                else:
                    bert_pooler = BertPooler(bert_dir)

        return what_type_of_bert(), bert, bert_pooler

    @staticmethod
    def _get_me_char_embeddings(embeddings: Dict[str, int], input_features: Tuple[str, ...]) \
            -> Dict[str, LstmSeq2VecEncoder]:
        return {
            cat: LstmSeq2VecEncoder(
                input_size=embeddings[cat],
                hidden_size=embeddings[f"{cat}_encoded"],
                num_layers=2,
                bidirectional=True,
                dropout=.3
            )
            for cat in {"token_char", "lemma_char"}
            if cat in input_features
        }

    @staticmethod
    def init_for_training_with_params(
        input_features: Tuple[str, ...] = ("token_subword",),
        bert_dir: str = "./bert/latin_bert",
        seq2vec_encoder_type: Seq2VecEncoderType = Seq2VecEncoderType.LSTM,
        model_class=FeatureEmbeddingClassifier,
        additional_model_kwargs: Dict[str, Any] = None,
        batches_per_epoch: Optional[int] = None,
        reader_kwargs: Dict[str, Any] = None,
        encoder_hidden_size: int = 64,
        agglomerate_msd: bool = False,
        use_bert_highway: bool = False,
        model_embedding_kwargs: Optional[Dict[str, Any]] = None,
        basis_vector_configuration: BasisVectorConfiguration = None,
    ) -> Tuple["Seligator", DatasetReader, DataLoader, DataLoader]:

        # For test, just change the input feature here
        # INPUT_FEATURES = ("token", "lemma", "token_char")  # , "token_subword")
        features_encoder = Seligator._get_seq2vec_encoder(
            seq2vec_encoder_type=seq2vec_encoder_type,
            encoder_hidden_size=encoder_hidden_size,
            basis_vector_configuration=basis_vector_configuration
        )
        use_bert = "token_subword" in input_features
        get_me_bert, bert_embedder, bert_pooler = Seligator._get_me_bert(
            use_bert,
            highway=use_bert_highway,
            bert_dir=bert_dir
        )
        model_embedding_kwargs = model_embedding_kwargs or {}
        embeddings = merge(
            model_embedding_kwargs.pop("emb_dims", {}),
            {k: v for k, v in EMBEDDING_DIMENSIONS.items()}
        )

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

        if input_features != reader.categories:  # eg. Might have been changed by agglomeration
            input_features = reader.categories

        embedding_encoders = Seligator._get_me_char_embeddings(embeddings, input_features=input_features)
        model = Seligator._initiate_model(
            model_class=model_class,
            vocab=vocab,
            input_features=input_features,
            mixed_encoder=MixedEmbeddingEncoder.build(
                vocabulary=vocab,
                emb_dims=MixedEmbeddingEncoder.merge_default_embeddings(embeddings),
                input_features=input_features,
                features_encoder=features_encoder,
                char_encoders=embedding_encoders,
                use_bert=use_bert,
                bert_embedder=bert_embedder,
                bert_pooler=bert_pooler,
                model_embedding_kwargs=model_embedding_kwargs
            ),
            basis_vector_configuration=basis_vector_configuration,
            **(additional_model_kwargs or {})
        )
        return Seligator(model), reader, train, dev

    @staticmethod
    def _initiate_model(model_class, **model_kwargs):
        return model_class(**model_kwargs)

    @classmethod
    def init_from_params(cls):
        return None


def train_and_get(model, train, dev, lr: float = 1e-4, use_cpu: bool = False,
                  **train_kwargs) -> FeatureEmbeddingClassifier:
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

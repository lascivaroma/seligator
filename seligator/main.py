from typing import Tuple, Optional, Dict, Any, Callable, Union, List
import copy
import json
import os

import torch
import torch.nn as nn

from allennlp.data import DataLoader, DatasetReader, Vocabulary, Instance
from allennlp.modules.seq2vec_encoders import LstmSeq2VecEncoder, BertPooler

from seligator.common.load_save import load, merge, CustomEncoder
from seligator.common.constants import EMBEDDING_DIMENSIONS
from seligator.common.params import Seq2VecEncoderType, BasisVectorConfiguration, MetadataEncoding
from seligator.common.bert_utils import what_type_of_bert, GetMeBert

from seligator.dataset.utils import get_fields
from seligator.dataset.readers import ClassificationTsvReader, XMLDatasetReader
from seligator.prediction import represent, simple_batcher

from seligator.models.classifier import FeatureEmbeddingClassifier
from seligator.modules.mixed_encoders import MixedEmbeddingEncoder
from seligator.training.trainer import generate_all_data, train_model
from seligator.modules.seq2vec import \
    HierarchicalAttentionalEncoder, PoolerHighway, SumAndLinear, MetadataEnrichedAttentionalLSTM

import logging

logging.getLogger().setLevel(logging.INFO)


class Seligator:
    def __init__(self, model, init_params: Dict[str, Any]):
        self.model: FeatureEmbeddingClassifier = model
        self.vocabulary: Vocabulary = self.model.vocab
        self._init_params = init_params

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
    def _get_me_bert(use: bool, highway: bool, bert_dir: str = None) -> Tuple[GetMeBert, nn.Module, nn.Module]:
        bert, bert_pooler = None, None
        if use:
            get_me_bert = what_type_of_bert(directory=bert_dir, trainable=False, hugginface=False)
            bert = get_me_bert.embedder
            if highway:
                bert_pooler = PoolerHighway(BertPooler(bert_dir), 256)
            else:
                bert_pooler = BertPooler(bert_dir)
            return get_me_bert, bert, bert_pooler
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
    def init_from_params(
        token_features: Tuple[str, ...] = ("token_subword",),
        msd_features: Tuple[str, ...] = (),
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
        training: bool = True,  # If set to true, return reader ands loader
        vocabulary_path: str = None,  # When training is False, requires vocabulary path
        folder: str = "dataset/main"

    ) -> Union["Seligator", Tuple["Seligator", DatasetReader, DataLoader, DataLoader]]:
        init_params = copy.deepcopy({
            "token_features": token_features,
            "msd_features": msd_features,
            "bert_dir": bert_dir,
            "seq2vec_encoder_type": seq2vec_encoder_type,
            "model_class": model_class,
            "additional_model_kwargs": additional_model_kwargs,
            "batches_per_epoch": batches_per_epoch,
            "reader_kwargs": reader_kwargs,
            "encoder_hidden_size": encoder_hidden_size,
            "agglomerate_msd": agglomerate_msd,
            "use_bert_highway": use_bert_highway,
            "model_embedding_kwargs": model_embedding_kwargs,
            "basis_vector_configuration": basis_vector_configuration
         })
        if not training and not vocabulary_path:
            raise ValueError("When not training, the `vocabulary_path` is required for initiating Seligator")
        # For test, just change the input feature here
        # INPUT_FEATURES = ("token", "lemma", "token_char")  # , "token_subword")
        features_encoder = Seligator._get_seq2vec_encoder(
            seq2vec_encoder_type=seq2vec_encoder_type,
            encoder_hidden_size=encoder_hidden_size,
            basis_vector_configuration=basis_vector_configuration
        )
        use_bert = "token_subword" in token_features
        get_me_bert, bert_embedder, bert_pooler = Seligator._get_me_bert(
            use=use_bert,
            highway=use_bert_highway,
            bert_dir=bert_dir
        )
        model_embedding_kwargs = model_embedding_kwargs or {}
        embeddings = merge(
            model_embedding_kwargs.pop("emb_dims", {}),
            {k: v for k, v in EMBEDDING_DIMENSIONS.items()}
        )

        if training:
            train, dev, vocab, reader = generate_all_data(
                token_features=token_features,
                msd_features=msd_features,
                get_me_bert=get_me_bert,
                instance_type=model_class.INSTANCE_TYPE,
                batches_per_epoch=batches_per_epoch,
                folder=folder,
                **{**(reader_kwargs or {}), "agglomerate_msd": agglomerate_msd}
            )
            input_features = reader.categories
        else:
            vocab = Vocabulary.from_files(vocabulary_path)
            input_features, agglomerate_msd = get_fields(token_features, msd_features, agglomerate_msd)
            model_embedding_kwargs.pop("pretrained_embeddings")
        if basis_vector_configuration:
            logging.info("Fitting the BasisVectorConfiguration")
            basis_vector_configuration.set_metadata_categories_dims(vocab)

        embedding_encoders = Seligator._get_me_char_embeddings(embeddings, input_features=input_features)
        model = model_class(
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
        if training:
            return Seligator(model, init_params=init_params), reader, train, dev
        else:
            return Seligator(model, init_params=init_params)

    def save_model(self, path: str = "./saved_model/"):
        self.model.vocab.save_to_files(os.path.join(path, "vocabulary"))
        self.model.save_to_file(os.path.join(path, "model.pth"))
        with open(os.path.join(path, "params.json"), "w") as j:
            json.dump(self._init_params, j, cls=CustomEncoder)

    @classmethod
    def load_model(cls, path: str):
        sel = cls.init_from_params(
            vocabulary_path=os.path.join(path, "vocabulary"),
            training=False,
            **load(os.path.join(path, "params.json"))
        )
        sel.model.load_state_dict(torch.load(os.path.join(path, "model.pth")))
        return sel

    def get_reader(self, cls=ClassificationTsvReader) -> Union[XMLDatasetReader, ClassificationTsvReader]:
        get_me_bert, bert_embedder, bert_pooler = Seligator._get_me_bert(
            use="token_subword" in self._init_params.get("token_features", []),
            highway=self._init_params.get("use_bert_highway"),
            bert_dir=self._init_params.get("bert_dir")
        )
        reader = cls(
            token_features=self._init_params.get("token_features"),
            msd_features=self._init_params.get("msd_features"),
            get_me_bert=get_me_bert,
            instance_type=self.model.INSTANCE_TYPE,
            **{
                **{
                    key: val
                    for key, val in self._init_params.get("reader_kwargs", {}).items()
                    if key not in {"ratio_train", "batch_size"}
                },
                "agglomerate_msd": self._init_params.get("agglomerate_msd")
            }

        )
        return reader

    def get_xml_loader(self, xml_file, metadata: Dict[str, Any] = None) -> List[Instance]:
        reader = self.get_reader(XMLDatasetReader)
        instances = list(reader.read_with_default_value(xml_file, default_metadata_tokens=[
            f"{k}={v}" for k, v in metadata.items()
        ]))
        return instances

    def predict_on_xml(self, xml_file, batch_size: int = 8, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        instances = self.get_xml_loader(xml_file, metadata=metadata or {})
        batcher = simple_batcher(list(instances), n=batch_size)

        predictions = []
        for batch in batcher:
            predictions.extend(self.model.forward_on_instances(batch))

        output = []
        for inst, pred in zip(instances, predictions):
            output.append(represent(instance=inst, prediction=pred, label_vocabulary=self.model.labels, is_gt=False))

        return output


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

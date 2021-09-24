from typing import Dict, Optional, Tuple, Any
import enum


@enum.unique
class MetadataEncoding(enum.Enum):
    IGNORE = 0
    AS_TOKEN = 1
    AS_CATEGORICAL = 2


def get_metadata_field_name(metadata_name: str) -> str:
    return f"metadata_categoricals_{metadata_name}"


def get_metadata_namespace(metadata_name: str) -> str:
    return metadata_name + "_ns_labels"


@enum.unique
class BertPoolerClass(enum.Enum):
    CLS = 0
    CLS_Highway = 1
    MEAN = 2
    MAX = 3
    MEANMAX = 4
    GRU = 5
    TOKEN_MERGE = 6
    HAN = 7
    EnrichedLSTM = 8
    EnrichedAttention = 9


@enum.unique
class Seq2VecEncoderType(enum.Enum):
    LSTM = 0
    HAN = 1
    AttentionPooling = 2
    MetadataAttentionPooling = 3
    MetadataLSTM = 4
    CONV = 5
    TDS = 6  # ToImplement
    GRU = 7
    BERT_ONLY = 8


class MetaParamManager:
    def __init__(self):
        self.meta_em = {}

    def state_dict(self):
        return self.meta_em

    def register(self, name, param):
        self.meta_em[name] = param


class BasisVectorConfiguration:
    def __init__(
            self,
            categories: Tuple[str, ...],
            param_manager: Optional[MetaParamManager] = None,
            emb_dim: int = 64,
            num_bases: int = 3,
            key_query_size: int = 64,
    ):
        self.categories_tuple = categories
        self.param_manager = param_manager or MetaParamManager()
        self.emb_dim = emb_dim
        self.num_bases = num_bases
        self.key_query_size: int = key_query_size
        self._categories = None

    def __repr__(self):
        return f"<BasisVectorConfiguration cats='{','.join(self.categories_tuple)}' />"

    def to_dict(self):
        return {
            "categories": self.categories_tuple,
            "categories_dim": self.categories if self._categories else None,
            "emb_dim": self.emb_dim,
            "num_bases": self.num_bases,
            "key_query_size": self.key_query_size
        }

    @classmethod
    def from_dict(cls, params):
        vocab_count = params.pop("categories_dim", None)
        x = cls(**params)
        x._categories = vocab_count
        return x

    @property
    def categories(self) -> Dict[str, int]:
        if self._categories:
            return self._categories
        raise ValueError("`set_metadata_categories_dim` was never run on BasisVectorConfiguration")

    def set_metadata_categories_dims(self, vocabulary):
        self._categories = {
            cat: vocabulary.get_vocab_size(get_metadata_namespace(cat))
            for cat in self.categories_tuple
        }

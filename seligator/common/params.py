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
class Seq2VecEncoderType(enum.Enum):
    LSTM = 0
    HAN = 1
    AttentionPooling = 2
    MetadataAttentionPooling = 3
    MetadataLSTM = 4


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

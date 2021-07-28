import enum


@enum.unique
class MetadataEncoding(enum.Enum):
    IGNORE = 0
    AS_TOKEN = 1
    AS_CATEGORICAL = 2


def get_metadata_field_name(metadata_name: str) -> str:
    return f"metadata_categoricals_{metadata_name}"


def get_metadata_namespace(metadata_name: str) -> str:
    return metadata_name+"_ns_labels"


@enum.unique
class Seq2VecEncoderType(enum.Enum):
    LSTM = 0
    HAN = 1
    AttentionPooling = 2
    MetadataAttentionPooling = 3

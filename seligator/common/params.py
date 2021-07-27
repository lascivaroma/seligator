import enum


@enum.unique
class MetadataEncoding(enum.Enum):
    IGNORE = 0
    AS_TOKEN = 1
    AS_CATEGORICAL = 2

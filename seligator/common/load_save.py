from .params import MetadataEncoding, Seq2VecEncoderType, BasisVectorConfiguration
from ..models import *
import json


def merge(new, default):
    """ Source = New , Destination = Default
    run me with nosetests --with-doctest file.py

    >>> a = { 'first' : { 'all_rows' : { 'pass' : 'dog', 'number' : '1' } } }
    >>> b = { 'first' : { 'all_rows' : { 'fail' : 'cat', 'number' : '5' } } }
    >>> merge(b, a) == { 'first' : { 'all_rows' : { 'pass' : 'dog', 'fail' : 'cat', 'number' : '5' } } }
    True
    """
    for key, value in new.items():
        if isinstance(value, dict):
            # get node or create one
            node = default.setdefault(key, {})
            merge(value, node)
        else:
            default[key] = value

    return default


class CustomEncoder(json.JSONEncoder):
    _PUBLIC_ENUMS = {
        "MetadataEncoding": MetadataEncoding,
        "Seq2VecEncoderType": Seq2VecEncoderType,
        # "BasisVectorConfiguration": BasisVectorConfiguration
    }
    _PUBLIC_CLASSES = {
        "SiameseClassifier": SiameseClassifier,
        "FeatureEmbeddingClassifier": FeatureEmbeddingClassifier,
        "TripletClassifier": TripletClassifier
    }

    def default(self, obj):
        if type(obj) in CustomEncoder._PUBLIC_ENUMS.values():
            return {"__enum__": str(obj)}
        elif isinstance(obj, type):
            if obj in CustomEncoder._PUBLIC_CLASSES.values():
                return {"__type__": str(obj.__name__)}
            else:
                print(obj)
        elif isinstance(obj, BasisVectorConfiguration):
            return {"__basis_vector_configuration__": obj.to_dict()}
        return json.JSONEncoder.default(self, obj)

    @staticmethod
    def object_hook(d):
        if "__enum__" in d:
            name, member = d["__enum__"].split(".")
            return getattr(CustomEncoder._PUBLIC_ENUMS[name], member)
        elif "__type__" in d:
            return CustomEncoder._PUBLIC_CLASSES[d["__type__"]]
        elif "__basis_vector_configuration__" in d:
            return BasisVectorConfiguration.from_dict(d["__basis_vector_configuration__"])
        else:
            return d


def load(iofile):
    if isinstance(iofile, str): # FilePath
        with open(iofile) as f:
            return json.load(f, object_hook=CustomEncoder.object_hook)
    else:
        raise ValueError("Unsupported input for loading configurations.")
# json.loads(json.dumps(get_kwargs(), cls=CustomEncoder), object_hook=CustomEncoder.object_hook)
from typing import Tuple, Optional
from dataclasses import dataclass

from allennlp.modules.token_embedders.pretrained_transformer_embedder import (
    PretrainedTransformerEmbedder, PretrainedTransformerTokenizer
)
from allennlp.data.token_indexers import PretrainedTransformerIndexer, TokenIndexer


@dataclass
class GetMeBert:
    """Class for keeping track of an item in inventory."""
    use_bert: bool = False
    embedder: PretrainedTransformerEmbedder = None
    indexer: TokenIndexer = None
    tokenizer: PretrainedTransformerTokenizer = None
    layer: int = -1


def what_type_of_bert(
        directory: Optional[str] = None,
        hugginface: bool = False,
        trainable: bool = False,
        layer: int = -1
) -> GetMeBert:
    if not directory:
        return GetMeBert(False)

    if hugginface:
        bert = PretrainedTransformerEmbedder(directory, train_parameters=trainable)
        tokenizer = PretrainedTransformerTokenizer(directory)
        return GetMeBert(True, bert, PretrainedTransformerIndexer(directory), tokenizer, layer)

    from seligator.modules.embedders.latinBert import LatinPretrainedTransformer
    from seligator.dataset.indexer import LatinSubwordTokenIndexer
    indexer = LatinSubwordTokenIndexer(vocab_path=directory, namespace="token_subword")
    tokenizer = indexer.tokenizer

    return GetMeBert(
        True,
        LatinPretrainedTransformer(directory, train_parameters=trainable, tokenizer=tokenizer, use_layer=layer),
        indexer,
        tokenizer,
        layer
    )

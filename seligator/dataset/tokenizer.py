from seligator.common.constants import CATS, BERT_DIR
from typing import List, Optional, Set, Dict
import logging

import regex as re

from tensor2tensor.data_generators.text_encoder import SubwordTextEncoder
from allennlp.data.tokenizers.token_class import Token
from allennlp.data.tokenizers import Tokenizer

from overrides import overrides

logger = logging.getLogger(__name__)



SPACES = re.compile(r"\s+")
APOSTROPHES = re.compile(r"[\"']+")


def latin_bert_normalization(string: str) -> str:
    return SPACES.sub(" ", APOSTROPHES.sub("", string.replace("\\", "").replace(";", "")).strip().lower())


class SubwordTextEncoderTokenizer(Tokenizer):
    # https://github.com/tensorflow/tensor2tensor/blob/78ba8019847426e988294fd58f8953d7990a8db7/tensor2tensor/data_generators/text_encoder.py#L448
    def __init__(self,
                 vocab: str,
                 add_special_tokens: bool = True,
                 max_length: Optional[int] = None,
                 special_tokens: Set[str] = None
                 ):
        self.vocab = vocab
        self._tokenizer = SubwordTextEncoder(vocab)
        # StartFix AttributeError: 'SubwordTextEncoder' object has no attribute '_cache_size'
        # ToDo: Find a way to avoid STE from tensor2tensor
        #self._tokenizer._cache_size = 2 ** 20
        #self._tokenizer._cache = [(None, None)] * self._tokenizer._cache_size
        # EndFix
        self._max_length = max_length
        self._add_special_tokens = add_special_tokens

        self._special_tokens = special_tokens or {"[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"}

        self.single_sequence_start_tokens = ["[CLS]"]
        self.single_sequence_start_tokens_ids = [self._tokenizer._subtoken_string_to_id["[CLS]"]]
        self.single_sequence_end_tokens = []#["[SEP]"]
        self.single_sequence_end_tokens_ids = []#[self._tokenizer._subtoken_string_to_id["[SEP]"]]

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def unk_token_id(self):
        return self._tokenizer._subtoken_string_to_id["[UNK]"]

    @property
    def pad_token_id(self):
        return self._tokenizer._subtoken_string_to_id["[PAD]"]

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._tokenizer._subtoken_string_to_id.get(token, self.unk_token_id)

    # ToDo: Check I did not miss any behavior here
    def build_inputs_with_special_tokens(self, segment):
        return segment

    def get_vocab(self) -> Dict[str, int]:
        return self._tokenizer._subtoken_string_to_id

    @property
    def id2tok(self) -> List[str]:
        return self._tokenizer._all_subtoken_strings

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.vocab_size

    @overrides
    def tokenize(self, text: str) -> List[Token]:
        """
        This method only handles a single sentence (or sequence) of text.
        """
        max_length = self._max_length
        if max_length is not None and not self._add_special_tokens:
            max_length += self.num_special_tokens_for_sequence()

        text = latin_bert_normalization(text)

        token_ids = self._tokenizer.encode(text)
        token_texts = self._tokenizer.decode_list(token_ids)

        if self._add_special_tokens:
            token_texts = [*self.single_sequence_start_tokens, *token_texts, *self.single_sequence_end_tokens]
            token_ids = [*self.single_sequence_start_tokens_ids, *token_ids, *self.single_sequence_end_tokens_ids]
            logger.debug(f"New text: {text}")

        special_tokens_mask = [1 if tok in self._special_tokens else 0 for tok in token_texts]

        tokens = []
        for token_id, token_text, special_token_mask in zip(
                token_ids, token_texts, special_tokens_mask
        ):
            if not self._add_special_tokens and special_token_mask == 1:
                continue

            tokens.append(
                Token(
                    text=token_text,
                    text_id=token_id,
                    type_id=None,
                    idx=None,
                    idx_end=None,
                )
            )

        return tokens


if __name__ == "__main__":
    tokenizer = SubwordTextEncoderTokenizer(f"{BERT_DIR}/latin_bert/vocab.txt")
    logger.setLevel(logging.DEBUG)
    z = tokenizer.tokenize("Nunc denique intellegimus quae desideranda in prioribus fuerint, postquam ea quae operta in ceteris veriti sumus in te reserata veneramur.")
    print(z)
    print(tokenizer.tokenize("lascivumque"))
    """
    from .tsv import ClassificationTsvReader
    from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer, TokenCharactersIndexer, \
        PretrainedTransformerIndexer
    from allennlp.data.tokenizers import Token, Tokenizer, WhitespaceTokenizer

    # Instantiate and use the dataset reader to read a file containing the data
    reader = ClassificationTsvReader(cats=("token", "token_subword"), token_indexers={
        "token": SingleIdTokenIndexer(namespace="token"),
        "token_subword": SingleIdTokenIndexer(namespace="token_subword")
    })
    dataset = list(reader.read("dataset/split/test.txt"))

    print("type of its first element: ", type(dataset[0]))
    print("size of dataset: ", len(dataset))
    print(dataset[0])
    """
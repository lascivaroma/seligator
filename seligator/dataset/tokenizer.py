from seligator.common.constants import CATS, BERT_DIR
from typing import List, Optional, Set, Dict, Union
import logging

import regex as re

from tensor2tensor.data_generators.text_encoder import SubwordTextEncoder
from allennlp.data.tokenizers.token_class import Token
from allennlp.data.tokenizers import Tokenizer, CharacterTokenizer

from overrides import overrides
from operator import attrgetter

logger = logging.getLogger(__name__)



SPACES = re.compile(r"\s+")
APOSTROPHES = re.compile(r"[\"']+")


def latin_bert_normalization(string: str) -> str:
    return SPACES.sub(" ", APOSTROPHES.sub("", string.replace("\\", "").replace(";", "")).strip().lower())


class ReworkedSTE(SubwordTextEncoder):
    def __init__(self, vocab):
        super(ReworkedSTE, self).__init__(vocab)

    def _token_to_subtoken_ids(self, token):
        return super(ReworkedSTE, self)._token_to_subtoken_ids(token.strip())


class BertToken(Token):
    def __init__(self, orig_token_id: int = None, **kwargs):
        super(BertToken, self).__init__(**kwargs)
        self.orig_token_id = orig_token_id


class LatinSubwordTextEncoderTokenizer(Tokenizer):
    # https://github.com/tensorflow/tensor2tensor/blob/78ba8019847426e988294fd58f8953d7990a8db7/tensor2tensor/data_generators/text_encoder.py#L448
    def __init__(self,
                 vocab: str,
                 add_special_tokens: bool = True,
                 max_length: Optional[int] = None,
                 special_tokens: Set[str] = None
                 ):
        self.vocab = vocab
        self._tokenizer = ReworkedSTE(vocab)
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
        self.single_sequence_end_tokens = ["[SEP]"]
        self.single_sequence_end_tokens_ids = [self._tokenizer._subtoken_string_to_id["[SEP]"]]

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
        # logging.info(text)

        token_ids = self._tokenizer.encode(text)
        token_texts = self._tokenizer.decode_list(token_ids)

        if self._add_special_tokens:
            token_texts = [*self.single_sequence_start_tokens, *token_texts, *self.single_sequence_end_tokens]
            token_ids = [*self.single_sequence_start_tokens_ids, *token_ids, *self.single_sequence_end_tokens_ids]
            # logger.debug(f"New text: {text}")

        special_tokens_mask = [1 if tok in self._special_tokens and tok != "[CLS]" else 0 for tok in token_texts]

        tokens = []
        _orig_token = 0
        for token_id, token_text, special_token_mask in zip(
                token_ids, token_texts, special_tokens_mask
        ):
            if not self._add_special_tokens and special_token_mask == 1:
                continue
            tokens.append(
                BertToken(
                    text=token_text,
                    text_id=token_id,
                    type_id=0,
                    idx=token_id,
                    idx_end=None,
                    orig_token_id=_orig_token if special_token_mask == 0 and token_text != "[CLS]" else -1
                )
            )
            if special_token_mask == 0 and token_text != "[CLS]" and token_text.endswith("_"):
                _orig_token += 1
        # logger.info(str(tokens))
        return tokens


class MultiTagFeatureTokenizer(CharacterTokenizer):
    def __init__(self,
                 features: Optional[List[str]] = None,
                 default_value: Optional[str] = "_",
                 start_tokens: List[Union[str, int]] = None,
                 end_tokens: List[Union[str, int]] = None):

        super(MultiTagFeatureTokenizer, self).__init__(
            byte_encoding=None,
            lowercase_characters=False,
            start_tokens=start_tokens,
            end_tokens=end_tokens)
        self._features = features or []
        self._default_value: str = default_value

    @property
    def default_token(self) -> Dict[str, str]:
        return {
            cat: self._default_value
            for cat in self._features
        }

    @overrides
    def tokenize(self, text: Dict[str, str]) -> List[Token]:
        tokens = sorted([
            Token(f"{cat}={feat}")
                for cat, feat in {
                    **self.default_token,
                    **text
                }.items()
            ],
            key=attrgetter("text")
        )
        for start_token in self._start_tokens:
            if isinstance(start_token, int):
                token = Token(text_id=start_token, idx=0)
            else:
                token = Token(text=start_token, idx=0)
            tokens.insert(0, token)
        for end_token in self._end_tokens:
            if isinstance(end_token, int):
                token = Token(text_id=end_token, idx=0)
            else:
                token = Token(text=end_token, idx=0)
            tokens.append(token)
        return tokens


if __name__ == "__main__":
    TEST_BERT = True
    TEST_Feature = False
    logger.setLevel(logging.DEBUG)
    """
    Non	non	ADVneg	-	-	-	-	-	-	-	-
tempore	tempus1	NOMcom	Abl	Sing	-	-	-	-	-	-
prisco	priscus	ADJqua	Abl	Sing	MascNeut	-	-	-	-	Pos
dicendum	dico2	VER	Nom	Sing	Neut	Adj	-	Pass	-	-
fuit	sum1	VER	-	Sing	-	Ind	Perf	Act	3	-
,	,	PUNC	-	-	-	-	-	-	-	-
sed	sed	CONcoo	-	-	-	-	-	-	-	-
:	:	PUNC	-	-	-	-	-	-	-	-
tempore	tempus1	NOMcom	Abl	Sing	-	-	-	-	-	-
,	,	PUNC	-	-	-	-	-	-	-	-
quod	quod1	PROrel	Nom	Sing	Neut	-	-	-	-	-
prisca	priscus	ADJqua	Abl	Sing	Fem	-	-	-	-	Pos
imitatur	imitor	VER	-	Sing	-	Ind	Pres	Dep	3	-
.	.	PUNC	-	-	-	-	-	-	-	-"""
    sentences = [[    2,  4235, 28847,    24,  7216,   116,    31,  1099, 23008,    24,
            15, 21314,    37,  4138,    10,  9911,    27, 18095,   311,    12,
          6190,  3099,   264, 17730,     3,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0],
        [    2,    15,  6854,  4153,     9,   180,  9451,  4096,  4299,     9,
          4134,  6195,   180,  1858,  5552,    24, 14163,   390,  9712,    14,
         12593, 26414,    83,     3,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0],
        [    2,    86,   107,   115, 22547,    34,    36, 23645, 12984,   874,
         10831,   116,   463, 11117, 12533, 21886,    85,  5194,  8107,    11,
            35,   823,    41, 20417,    19,  4411, 26237, 12434,   496,   406,
          4434,  7268,    31,   155,  4275, 22046,    77,     3],
        [    2,    20, 13638,  8721,    59, 25765, 26077, 21374,    19, 16684,
             9,   704,     3,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0]]
    if TEST_BERT:
        tokenizer = LatinSubwordTextEncoderTokenizer(f"{BERT_DIR}/vocab.txt")
        z = tokenizer.tokenize("non tempore prisco dicendum fuit , sed : tempore , quod prisca imitatur .")
        #print(z)
        print([tok.text_id for tok in z])
        print([tok.orig_token_id for tok in z])
        #print(tokenizer.convert_tokens_to_ids(tokenizer.tokenize("lascivumque")))
        for sent in sentences:
            print(tokenizer._tokenizer.decode_list(sent))

    if TEST_Feature:
        tokenizer = MultiTagFeatureTokenizer(["pos", "tense", "gend"])
        print(tokenizer.tokenize({"gend": "MascFem"}))



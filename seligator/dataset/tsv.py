from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import TextField, LabelField, Field
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer, TokenCharactersIndexer, \
    PretrainedTransformerIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WhitespaceTokenizer

from seligator.common.constants import CATS, BERT_DIR
from seligator.dataset.indexer import SubwordTokenIndexer
from typing import Dict, Iterable, Tuple, List, Optional

import logging
import os


def build_token_indexers(cats: Iterable[str] = None, bert_dir: Optional[str] = None) -> Dict[str, TokenIndexer]:
    if not cats:
        cats = CATS

    def get_indexer(category: str) -> TokenIndexer:
        if category.endswith("_char"):
            return TokenCharactersIndexer(namespace=category)
        elif category.endswith("_subword"):
            return SubwordTokenIndexer(namespace=category, vocab_path=bert_dir)
        else:
            return SingleIdTokenIndexer(namespace=category)

    return {
        task.lower(): get_indexer(task.lower())
        for task in cats
    }


@DatasetReader.register("classification-tsv")
class ClassificationTsvReader(DatasetReader):
    def __init__(
            self,
            tokenizer: Tokenizer = None,
            token_indexers: Dict[str, TokenIndexer] = None,
            max_tokens: int = None,
            cats: Tuple[str, ...] = CATS,
            use_only: Tuple[str, ...] = None,
            bert_dir: Optional[str] = None,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.features: Tuple[str, ...] = cats
        self.categories: Tuple[str, ...] = ("token", "lemma", "token_char", "pos", *self.features)

        if use_only:
            self.categories = use_only

        logging.info(f"Dataset reader set with following categories: {', '.join(self.categories)}")
        self.tokenizer = tokenizer or WhitespaceTokenizer()
        self.token_indexers = token_indexers or build_token_indexers(cats=self.categories, bert_dir=bert_dir)
        logging.info(f"Indexer set for following categories: {', '.join(self.token_indexers.keys())}")
        self.max_tokens = max_tokens

    def text_to_instance(self, content: List[Dict[str, str]], label: str = None) -> Instance:
        """ Parse the output of content into

        """
        fields: Dict[str, List[Token]] = {cat: [] for cat in self.categories}
        if "token_subword" in fields:
            normalized = " ".join([tok["token"] for tok in content])
            try:
                fields["token_subword"].extend(self.token_indexers["token_subword"].tokenizer.tokenize(normalized))
            except AssertionError:
                logging.error(f"Error on {normalized}")
                raise
        for token_repr in content:
            for cat, value in token_repr.items():
                if cat in fields:
                    fields[cat].append(Token(value))
                if cat == "token" and "token_char" in self.categories:
                    fields["token_char"].append(Token(value))

        if self.max_tokens:
            fields = {cat: fields[cat][:self.max_tokens] for cat in fields}

        #if "token_char" in self.categories:
        #    fields["token_char"] = [] + fields["token"]

        fields: Dict[str, Field] = {
            cat.lower(): TextField(fields[cat], token_indexers={cat: self.token_indexers[cat]})
            for cat in fields
        }

        if label:
            fields["label"] = LabelField(label)

        return Instance(fields)

    def _read(self, file_path: str) -> Iterable[Instance]:
        """ Reads a file into Instances

        The dataset should look like the following:
        - The first line should contain the header, starting with [header] and using TAB as a delimiter.
        - Samples should starts with #[TAG] and be followed by its label
        - Samples should be separated by at least two empty lines
        - Samples tokens and tags should be described by the column in the [header], ignoring the fact it's missing
            a column

        Example:

        ```tsv
        [header]	token	lemma	pos	Case	Numb	Gend	Mood	Tense	Voice	Person	Deg
        #[TAG]negative
        Cyrenis	Cyrenae	NOMpro	Abl	Plur	-	-	-	-	-	-
        liberalem	liberalis	ADJqua	Acc	Sing	MascFem	-	-	-	-	Pos
        in	in	PRE	-	-	-	-	-	-	-	-
        publicanos	publicanus1	NOMcom	Acc	Plur	-	-	-	-	-	-
        ,	,	PUNC	-	-	-	-	-	-	-	-
        iustum	iustus	ADJqua	Acc	Sing	MascNeut	-	-	-	-	Pos
        in	in	PRE	-	-	-	-	-	-	-	-
        socios	socius1	NOMcom	Acc	Plur	-	-	-	-	-	-
        fuisse	sum1	VER	-	-	-	Inf	Perf	Act	-	-
        .	.	PUNC	-	-	-	-	-	-	-	-
        ```

        """
        with open(file_path, "r") as lines:
            header = []
            content = []
            label = None
            for idx, line in enumerate(lines):
                line = line.strip()
                # First line is the header in our files
                # idx[0] should start with [header]
                if idx == 0:
                    if not line.startswith("[header]"):
                        raise ValueError(f"File {file_path} does not start with a header")
                    _, *header = line.strip().split("\t")
                    continue
                # If a line is empty, we are jumping to another sample.
                #   Multiple empty lines can follow each other, hence checking content
                elif not line:
                    if content:
                        if not label:
                            raise ValueError("A label was not found")
                        yield self.text_to_instance(content, label)
                        content = []
                        label = None
                    continue
                elif line.startswith("#[TAG]"):
                    label = line.replace("#[TAG]", "")
                else:
                    content.append(dict(zip(header, line.split("\t"))))


def build_dataset_reader(cats: Tuple[str, ...] = CATS, use_only: Tuple[str, ...] = None,
                         bert_dir: Optional[str] = None) -> DatasetReader:
    return ClassificationTsvReader(cats=cats, use_only=use_only, bert_dir=bert_dir)


if __name__ == "__main__":
    # Instantiate and use the dataset reader to read a file containing the data
    reader = ClassificationTsvReader(cats=CATS)
    dataset = list(reader.read("dataset/split/test.txt"))

    print("type of its first element: ", type(dataset[0]))
    print("size of dataset: ", len(dataset))
    print(dataset[0])

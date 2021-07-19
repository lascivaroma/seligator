from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import TextField, LabelField, Field
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer, TokenCharactersIndexer, \
    PretrainedTransformerIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WhitespaceTokenizer

from seligator.common.constants import CATS, BERT_DIR
from seligator.dataset.indexer import LatinSubwordTokenIndexer
from seligator.common.bert_utils import GetMeBert
from typing import Dict, Iterable, Tuple, List, Optional, Any

import logging
import os

from collections import defaultdict
import random


def build_token_indexers(
        cats: Iterable[str] = None,
        get_me_bert: Optional[GetMeBert] = None
) -> Dict[str, TokenIndexer]:
    if not cats:
        cats = CATS

    def get_indexer(category: str) -> TokenIndexer:
        if category.endswith("_char"):
            return TokenCharactersIndexer(namespace=category)
        elif category.endswith("_subword"):
            if get_me_bert and get_me_bert.use_bert:
                return get_me_bert.indexer
            raise Exception("GetMeBert was not set !")
        else:
            return SingleIdTokenIndexer(namespace=category)

    return {
        task.lower(): get_indexer(task.lower())
        for task in cats
    }


@DatasetReader.register("classification-tsv")
class ClassificationTsvReader(DatasetReader):
    INSTANCE_TYPES = {"default", "siamese", "triplet"}
    def __init__(
            self,
            tokenizer: Tokenizer = None,
            token_indexers: Dict[str, TokenIndexer] = None,
            max_tokens: int = 256,
            cats: Tuple[str, ...] = CATS,
            input_features: Tuple[str, ...] = None,
            get_me_bert: Optional[GetMeBert] = GetMeBert(),
            instance_type: str = "default",
            siamese_probability: float = 1.0,
            siamese_samples: Dict[str, List[Dict[str, Any]]] = None,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.features: Tuple[str, ...] = cats
        self.categories: Tuple[str, ...] = ("token", "lemma", "token_char", "pos", *self.features)

        if input_features:
            self.categories = input_features

        logging.info(f"Dataset reader set with following categories: {', '.join(self.categories)}")
        self.tokenizer = tokenizer or WhitespaceTokenizer()
        self.token_indexers = token_indexers or build_token_indexers(cats=self.categories, get_me_bert=get_me_bert)
        self.bert_tokenizer = get_me_bert.tokenizer
        logging.info(f"Indexer set for following categories: {', '.join(self.token_indexers.keys())}")
        self.max_tokens = max_tokens

        # If Siamese is true, the first sentence that is positive will be set as the example
        #   The second one as well

        if instance_type.lower() not in {"default", "siamese", "triplet"}:
            raise ValueError("`instance_type` must be one of " + str({"default", "siamese", "triplet"}))

        self.instance_type: str = instance_type.lower()
        self.siamese_probability: float = siamese_probability
        self.siamese_samples: Dict[str, Instance] = siamese_samples or {}
        if self.instance_type in {"siamese", "triplet"}:
            logging.info(f"Siamese Models for positive: {len(self.siamese_samples['positive'])}")
            logging.info(f"Siamese Models for negative: {len(self.siamese_samples['negative'])}")

    def text_to_instance(self, content: List[Dict[str, str]], label: str = None) -> Instance:
        """ Parse the output of content into

        """
        fields: Dict[str, List[Token]] = {cat: [] for cat in self.categories}
        if "token_subword" in fields:
            normalized = " ".join([tok["token"] for tok in content if tok["token"][0] != "{"])
            try:
                fields["token_subword"].extend(self.bert_tokenizer.tokenize(normalized))
            except AssertionError:
                logging.error(f"Error on {normalized}")
                raise

        for token_repr in content:
            for cat, value in token_repr.items():
                if cat in fields:
                    fields[cat].append(Token(value))
                if cat == "token" and "token_char" in self.categories:
                    fields["token_char"].append(Token(value))
                if cat == "lemma" and "lemma_char" in self.categories:
                    fields["lemma_char"].append(Token(value))

        if self.max_tokens:
            fields = {cat: fields[cat][:self.max_tokens] for cat in fields}

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
        sentences: List[Instance] = []

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
                    _, *header = line.strip().lower().split("\t")
                    continue
                # If a line is empty, we are jumping to another sample.
                #   Multiple empty lines can follow each other, hence checking content
                elif not line:
                    if content:
                        if not label:
                            raise ValueError("A label was not found")

                        # If we are not in siamese, yield
                        if self.instance_type == "default":
                            yield self.text_to_instance(content, label)
                        else:
                            # We check that positive and negative has been set
                            s = self.text_to_instance(content, label)
                            sentences.append(s)
                        content = []
                        label = None
                    continue
                elif line.startswith("#[TAG]"):
                    label = line.replace("#[TAG]", "").strip()
                else:
                    content.append(dict(zip(header, line.split("\t"))))

        if content:
            if self.instance_type == "default":
                yield self.text_to_instance(content, label)
            else:
                s = self.text_to_instance(content, label)
                sentences.append(s)

        if self.instance_type == "triplet":
            for sentence in sentences:
                pos, neg = random.choice(self.siamese_samples["positive"]), \
                           random.choice(self.siamese_samples["negative"])
                yield Instance(
                    {
                        **{f"left_{key}": value for key, value in sentence.items()},
                        **{f"positive_{key}": value for key, value in pos.items()},
                        **{f"negative_{key}": value for key, value in neg.items()}
                    }
                )
        elif self.instance_type == "siamese":
            for sentence in sentences:
                if len(self.siamese_samples["negative"]) > 0:
                    pooled_label: str = "positive" if random.random() < self.siamese_probability else "negative"
                else:
                    pooled_label = "positive"
                right = random.choice(self.siamese_samples[pooled_label])
                yield Instance(
                    {
                        **{f"left_{key}": value for key, value in sentence.items()},
                        **{f"right_{key}": value for key, value in right.items()}
                    }
                )


def get_siamese_samples(
        reader: ClassificationTsvReader, siamese_filepath: str = "dataset/split/siamese.txt"
                        ) -> Dict[str, List[Dict[str, Any]]]:
    logging.info("Reading Siamese Samples")
    siamese_data = {"positive": [], "negative": []}

    for instance in reader.read(siamese_filepath):
        siamese_data[instance.human_readable_dict()["label"]].append(instance.fields)

    return siamese_data


if __name__ == "__main__":
    # Instantiate and use the dataset reader to read a file containing the data
    reader = ClassificationTsvReader(cats=CATS, input_features=("token", "lemma", "pos", "tense"),# siamese=True,
                                     bert_dir="./bert/latin_bert")
    dataset = list(reader.read("dataset/split/train.txt"))

    print("type of its first element: ", type(dataset[0]))
    print("size of dataset: ", len(dataset))
    print(dataset[0])

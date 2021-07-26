import logging
import random
from typing import Dict, Iterable, Tuple, List, Optional, Any, Set, Union, ClassVar

from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import TextField, LabelField, Field, MultiLabelField, ListField, MetadataField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer, TokenCharactersIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WhitespaceTokenizer

from seligator.common.constants import CATS, MSD_CAT_NAME
# from seligator.dataset.indexer import LatinSubwordTokenIndexer, MultipleFeatureVectorIndexer
from seligator.common.bert_utils import GetMeBert


# ToDo: Add MetadataField to keep the sentence for example, the author and other stuff


def what_kind_of_field(category: str, values: List, token_indexer: Optional[TokenIndexer]) -> ClassVar["Field"]:
    if category == MSD_CAT_NAME:
        return ListField(values)
    return TextField(values, token_indexers={category: token_indexer})


def build_token_indexers(
        cats: Iterable[str] = None,
        get_me_bert: Optional[GetMeBert] = None,
        msd: Optional[Set[str]] = None
) -> Dict[str, TokenIndexer]:
    cats = cats or CATS

    def get_indexer(category: str) -> TokenIndexer:
        if category.endswith("_char"):
            return TokenCharactersIndexer(namespace=category)
        elif category.endswith("_subword"):
            if get_me_bert and get_me_bert.use_bert:
                return get_me_bert.indexer
            raise Exception("GetMeBert was not set !")
        else:
            return SingleIdTokenIndexer(namespace=category)

    if msd:
        return {
            #MSD_CAT_NAME: MultipleFeatureVectorIndexer(
            #    namespace=MSD_CAT_NAME,
            #    msd=msd or []
            #),
            **build_token_indexers(
                [cat for cat in cats if cat not in msd],
                get_me_bert=get_me_bert
            )
        }

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
            agglomerate_msd: bool = False,
            get_me_bert: Optional[GetMeBert] = GetMeBert(),
            instance_type: str = "default",
            siamese_probability: float = 1.0,
            siamese_samples: Dict[str, List[Dict[str, Any]]] = None,
            metadata_token_as_lemma_and_token: bool = False,
            **kwargs
    ):
        """

        :param tokenizer: Tokenizer to use
        :param token_indexers: Dict of input_key -> indexers
        :param max_tokens: Maximum amount of tokens per "sentence"
        :param cats: List of known token-features
        :param input_features: List of token-features to user
        :param agglomerate_msd: Instead of encoding each feature in its own namespace, all morpho-syntactical
        features (!= lemma, token, *_char, *_subword) are registered in a single namespace and agglutinated.
        :param get_me_bert: Information about bert usage
        :param instance_type: Type of instance to use ("default", "siamese", "triplet")
        :param siamese_probability: Probability to train against a positive example of siamese
        :param siamese_samples: Samples to train against for "siamese" and "triple" instance types.
        """
        super().__init__(**kwargs)

        self.features: Tuple[str, ...] = cats
        self.categories: Tuple[str, ...] = ("token", "lemma", "token_char", "pos", *self.features)

        if input_features:
            self.categories = input_features

        self._msd: Set[str] = {
            feat
            for feat in self.categories
            if "token" not in feat and "lemma" not in feat
        }
        self.agglomerate_msd: bool = agglomerate_msd
        if self.agglomerate_msd:
            self.categories = (
                MSD_CAT_NAME,
                *(
                   cat
                   for cat in self.categories
                   if cat not in self._msd
                )
            )

        logging.info(f"Dataset reader set with following categories: {', '.join(self.categories)}")
        self.tokenizer = tokenizer or WhitespaceTokenizer()
        self.token_indexers = token_indexers or build_token_indexers(
            cats=self.categories,
            get_me_bert=get_me_bert,
            msd=self._msd
        )
        self.bert_tokenizer = get_me_bert.tokenizer
        logging.info(f"Indexer set for following categories: {', '.join(self.token_indexers.keys())}")
        self.max_tokens = max_tokens

        # If Siamese is true, the first sentence that is positive will be set as the example
        #   The second one as well

        if instance_type.lower() not in self.INSTANCE_TYPES:
            raise ValueError("`instance_type` must be one of " + str(self.INSTANCE_TYPES))

        self.instance_type: str = instance_type.lower()
        self.siamese_probability: float = siamese_probability
        self.siamese_samples: Dict[str, Instance] = siamese_samples or {}
        if self.instance_type in {"siamese", "triplet"}:
            logging.info(f"Siamese Models for positive: {len(self.siamese_samples['positive'])}")
            logging.info(f"Siamese Models for negative: {len(self.siamese_samples['negative'])}")

        self.metadata_token_as_lemma_and_token: bool = metadata_token_as_lemma_and_token
        print("metadata_token_as_lemma_and_token", metadata_token_as_lemma_and_token)

    def text_to_instance(self,
                         content: List[Dict[str, str]],
                         label: str = None,
                         metadata_tokens: List[str] = None,
                         metadata_generic: Dict[str, str] = None
                         ) -> Instance:
        """ Parse the output of content into

        """
        sentence = []
        fields: Dict[str, List[Union[Token, MultiLabelField]]] = {cat: [] for cat in self.categories}
        if "token_subword" in fields:
            normalized = " ".join([
                tok["token"]
                for tok in content if tok["token"][0] != "{"
            ])
            try:
                fields["token_subword"].extend(self.bert_tokenizer.tokenize(normalized))
            except AssertionError:
                logging.error(f"Error on {normalized}")
                raise

        sentence = []

        # For each token in a sentence
        for token_repr in content:
            msd = {}
            for cat, value in token_repr.items():
                if cat in fields:
                    fields[cat].append(Token(value))
                if cat+"_char" in self.categories:
                    fields[cat+"_char"].append(Token(value))
                if self.agglomerate_msd and cat in self._msd:
                    msd[cat] = value
                if cat == "token":
                    sentence.append(value)
            if self.agglomerate_msd:
                fields[MSD_CAT_NAME].append(
                    MultiLabelField(
                        [f"{cat}:{val}" for cat, val in msd.items()],
                        label_namespace=MSD_CAT_NAME
                    )
                )

        if self.metadata_token_as_lemma_and_token:
            for cat in fields:
                if cat in {"token", "lemma"}:
                    fields[cat] = [Token(met) for met in sorted(metadata_tokens or [])] + fields[cat]
                elif cat == MSD_CAT_NAME:
                    fields[cat] = [
                                      MultiLabelField([], label_namespace=MSD_CAT_NAME) for _ in metadata_tokens or []
                                  ] + fields[cat]
                else:
                    fields[cat] = [
                                      Token("") for _ in metadata_tokens or []
                                  ] + fields[cat]

        if self.max_tokens:
            fields = {cat: fields[cat][:self.max_tokens] for cat in fields}

        fields: Dict[str, Field] = {
            cat.lower(): what_kind_of_field(cat, fields[cat], token_indexer=self.token_indexers.get(cat))
            for cat in fields
        }

        if label:
            fields["label"] = LabelField(label)

        fields["metadata"] = MetadataField({
            "sentence": (metadata_tokens or []) + sentence if self.metadata_token_as_lemma_and_token else sentence,
             **(metadata_generic or {})
        })

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

        content = []
        label = None
        metadatas = {}
        token_metadatas = []

        with open(file_path, "r") as lines:
            header = []
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
                            yield self.text_to_instance(
                                content, label,
                                metadata_generic=metadatas, metadata_tokens=token_metadatas
                            )
                        else:
                            # We check that positive and negative has been set
                            s = self.text_to_instance(
                                content, label,
                                metadata_generic=metadatas, metadata_tokens=token_metadatas
                            )
                            sentences.append(s)
                        content = []
                        label = None
                        metadatas = {}
                        token_metadatas = []
                    continue
                elif line.startswith("#[TAG]"):
                    label = line.replace("#[TAG]", "").strip()
                elif line.startswith("[GENERIC-METADATA"):
                    _, key, val = line.replace("[GENERIC-METADATA", "").replace("]", "").strip().split("---")
                    metadatas[key] = val
                elif line.startswith("[TOKEN-METADATA]"):
                    tok = line.replace("[TOKEN-METADATA]", "").strip()
                    token_metadatas.append(tok)
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
    reader = ClassificationTsvReader(
        cats=CATS,
        input_features=("token", "lemma", "pos", "tense"),  # siamese=True,
        # bert_dir="./bert/latin_bert",
        agglomerate_msd=True
    )
    dataset = list(reader.read("dataset/split/test.txt"))

    print("type of its first element: ", type(dataset[0]))
    print("size of dataset: ", len(dataset))
    print(dataset[0])

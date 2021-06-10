from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import TextField, LabelField, Field
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer, TokenCharactersIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WhitespaceTokenizer

from seligator.common.constants import CATS
from typing import Dict, Iterable, Tuple, List


def build_token_indexers(cats: Iterable[str] = CATS) -> Dict[str, TokenIndexer]:
    return {
        "token": SingleIdTokenIndexer(namespace="token"),
        "lemma": SingleIdTokenIndexer(namespace="lemma"),
        "token_char": TokenCharactersIndexer(namespace="token_char"),
        "pos": SingleIdTokenIndexer(namespace="pos"),
        **{
            task.lower(): SingleIdTokenIndexer(namespace=f"{task.lower()}")
            for task in cats
        }
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
            **kwargs
    ):
        super().__init__(**kwargs)

        self.features: Tuple[str, ...] = cats
        self.categories: Tuple[str, ...] = ("token", "lemma", "token_char", "pos", *self.features)

        if use_only:
            self.categories = use_only

        self.tokenizer = tokenizer or WhitespaceTokenizer()
        self.token_indexers = token_indexers or build_token_indexers(cats=cats)
        self.max_tokens = max_tokens

    def text_to_instance(self, content: List[Dict[str, str]], label: str = None) -> Instance:
        """ Parse the output of content into

        """
        fields: Dict[str, List[Token]] = {cat: [] for cat in self.categories}

        for token_repr in content:
            for cat, value in token_repr.items():
                if cat in fields:
                    fields[cat].append(Token(value))
                if cat == "token" and "token_char" in self.categories:
                    fields[cat].append(Token(value))

        if self.max_tokens:
            fields = {cat: fields[cat][:self.max_tokens] for cat in fields}

        if "token_char" in self.categories:
            fields["token_char"] = [] + fields["token"]

        fields: Dict[str, Field] = {
            cat.lower(): TextField(fields[cat], token_indexers=self.token_indexers)
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


def build_dataset_reader(cats: Tuple[str, ...] = CATS, use_only: Tuple[str, ...] = None) -> DatasetReader:
    return ClassificationTsvReader(cats=cats, use_only=use_only)


if __name__ == "__main__":
    # Instantiate and use the dataset reader to read a file containing the data
    reader = ClassificationTsvReader(cats=CATS)
    dataset = list(reader.read("dataset/split/test.txt"))

    print("type of its first element: ", type(dataset[0]))
    print("size of dataset: ", len(dataset))
    print(dataset[0])

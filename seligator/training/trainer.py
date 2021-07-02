import tempfile
import math
import logging
from typing import Tuple, List, Iterable, Callable, Union, Optional, Dict, Any
from allennlp.data import (
    DataLoader,
    DatasetReader,
    Instance,
    Vocabulary,
)
from allennlp.data.data_loaders import SimpleDataLoader
from allennlp.models import Model
from allennlp.training.trainer import Trainer
from allennlp.training.gradient_descent_trainer import GradientDescentTrainer
from allennlp.training.optimizers import AdamOptimizer


from seligator.dataset.tsv import ClassificationTsvReader, get_siamese_samples
from seligator.models.base import BaseModel


def read_data(reader: DatasetReader, use_siamese: bool = True) -> Tuple[List[Instance], List[Instance]]:
    """ Generates datasets

    :param reader: Dataset reader to parse the file
    :param use_siamese: Use the siamese data and merge it into train.txt
    """
    logging.info("Reading data")
    training_data = list(reader.read("dataset/split/train.txt"))
    if use_siamese:
        training_data.extend(list(reader.read("dataset/split/siamese.txt")))
    validation_data = list(reader.read("dataset/split/dev.txt"))

    return training_data, validation_data


def build_vocab(instances: Iterable[Instance]) -> Vocabulary:
    logging.info("Building the vocabulary")
    return Vocabulary.from_instances(instances)


def build_data_loaders(
    train_data: List[Instance],
    dev_data: List[Instance],
    batch_size: int = 8,
    batches_per_epoch: Optional[int] = None
) -> Tuple[DataLoader, DataLoader]:

    train_loader = SimpleDataLoader(
        train_data, batch_size, shuffle=True,
        batches_per_epoch=batches_per_epoch
    )
    dev_loader = SimpleDataLoader(
        dev_data, batch_size,
        shuffle=False,
        batches_per_epoch=batches_per_epoch
    )
    return train_loader, dev_loader


def build_trainer(
    model: BaseModel,
    serialization_dir: str,
    train_loader: DataLoader,
    dev_loader: DataLoader,
    cuda_device: Union[int, List[int]],
    patience: int = 2,
    num_epochs: int = 5,
    lr: float = 0.001
) -> Trainer:
    parameters = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    optimizer = AdamOptimizer(parameters, lr=lr)  # type: ignore
    logging.info(f"Num epochs: {num_epochs}")

    trainer = GradientDescentTrainer(
        model=model,
        serialization_dir=serialization_dir,
        data_loader=train_loader,
        validation_metric="+sim-accuracy" if model.IS_SIAMESE else "+fscore",
        validation_data_loader=dev_loader,
        num_epochs=num_epochs,
        patience=patience,
        optimizer=optimizer,
        cuda_device=cuda_device,
        use_amp=True
    )
    return trainer


def generate_all_data(
    input_features: Tuple[str, ...] = None,
    ratio_train: float = 1.0,
    batch_size: int = 16,
    batches_per_epoch: Optional[int] = None,
    bert_dir: Optional[str] = None,
    is_siamese: bool = False
) -> Tuple[DataLoader, DataLoader, Vocabulary, ClassificationTsvReader]:
    # Expects a siamese.txt file in the same directory as train.txt
    if is_siamese:
        # Samples are not siamese loaded. Siamese affects the other datasets
        siamese_reader = ClassificationTsvReader(
            input_features=input_features, bert_dir=bert_dir, siamese=False
        )
        siamese_samples = get_siamese_samples(siamese_reader)

        # Then we create the normal one
        dataset_reader = ClassificationTsvReader(
            input_features=input_features, bert_dir=bert_dir,
            siamese=True, siamese_samples=siamese_samples,
            token_indexers=siamese_reader.token_indexers,
            tokenizer=siamese_reader.tokenizer
        )
        # And parse the original data
        train_data, dev_data = read_data(dataset_reader, use_siamese=False)
    else:
        dataset_reader = ClassificationTsvReader(
            input_features=input_features, bert_dir=bert_dir, siamese=False
        )
        train_data, dev_data = read_data(dataset_reader, use_siamese=True)

    if ratio_train != 1.0:
        train_data = train_data[:math.ceil(ratio_train*len(train_data))]

    vocab = build_vocab(train_data)

    train_loader, dev_loader = build_data_loaders(
        train_data, dev_data,
        batch_size=batch_size, batches_per_epoch=batches_per_epoch
    )
    train_loader.index_with(vocab)
    dev_loader.index_with(vocab)

    return train_loader, dev_loader, vocab, dataset_reader

"""


    if "token_subword" in use_only:
        model = build_model(
            vocab=vocab, use_only=use_only,
            bert_dir=bert_dir, bert_tokenizer=dataset_reader.token_indexers["token_subword"].tokenizer)
    else:
        model = build_model(vocab=vocab, input_f=use_only)

    if cuda_device != -1:
        model = model.cuda()
        """


def train_model(
    model: BaseModel,
    train_loader: DataLoader,
    dev_loader: DataLoader,
    cuda_device: Union[List[int], int] = -1,
    patience: int = 2,
    num_epochs: int = 5
) :
    # You obviously won't want to create a temporary file for your training
    # results, but for execution in binder for this guide, we need to do this.
    print(f"---> Epochs:   {num_epochs}")
    print(f"---> Patience: {patience}")
    with tempfile.TemporaryDirectory() as serialization_dir:
        trainer = build_trainer(
            model,
            serialization_dir,
            train_loader, dev_loader,
            cuda_device=cuda_device,
            num_epochs=num_epochs,
            patience=patience
        )
        logging.info("Starting training")
        trainer.train()
        logging.info("Finished training")
    return


if __name__ == "__main__":
    from seligator.models import classifier
    #model, dataset_reader = run_training_loop(
    #    build_model=baseline.build_model,
    #    cuda_device=-1,
    #    use_only=("token", ),
    #    num_epochs=100,
    #    batch_size=16,
    #    patience=None
    #)
    import logging
    logging.getLogger().setLevel(logging.INFO)
    use_only = ("token_subword", )
    bert_dir = "./bert/latin_bert"

    siamese_reader = ClassificationTsvReader(
        use_only=use_only, bert_dir=bert_dir,
        siamese=False
    )
    siamese_samples = get_siamese_samples(siamese_reader)
    # Then we create the normal one
    dataset_reader = ClassificationTsvReader(
        use_only=use_only, bert_dir=bert_dir,
        siamese=True, siamese_samples=siamese_samples,
        token_indexers=siamese_reader.token_indexers,
        tokenizer=siamese_reader.tokenizer
    )
    train_data, dev_data = read_data(dataset_reader, use_siamese=False)

    model, dataset_reader = run_training_loop(
        build_model=lambda *a, **w: None,
        cuda_device=0,
        batch_size=4,
        batches_per_epoch=80,
        ratio_train=1,
        bert_dir="bert/latin_bert",
        use_only=("token_subword",),
        siamese=True
    )

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
from allennlp.training.optimizers import AdamOptimizer, AdamWOptimizer, AdadeltaOptimizer


from seligator.common.params import MetadataEncoding, get_metadata_namespace
from seligator.dataset.readers import ClassificationTsvReader, get_siamese_samples
from seligator.models.base import BaseModel
from seligator.common.bert_utils import GetMeBert


def read_data(reader: DatasetReader, use_siamese_set: bool = True,
              folder: str = "dataset/main") -> Tuple[List[Instance], List[Instance]]:
    """ Generates datasets

    :param reader: Dataset reader to parse the file
    :param use_siamese_set: Use the siamese data and merge it into train.txt
    """
    logging.info("Reading data")
    training_data = list(reader.read(f"{folder}/train.txt"))
    if use_siamese_set:
        training_data.extend(list(reader.read(f"{folder}/siamese.txt")))
    validation_data = list(reader.read(f"{folder}/dev.txt"))

    return training_data, validation_data


def build_vocab(instances: Iterable[Instance], add_unknown: Iterable[str] = None) -> Vocabulary:
    logging.info("Building the vocabulary")
    voc = Vocabulary.from_instances(instances)

    logging.warning(f"Adding OOV to following fields: {', '.join(add_unknown)}")
    for field in (add_unknown or []):
        voc.add_token_to_namespace(voc._oov_token, field)
    return voc


def build_data_loaders(
    train_data: List[Instance],
    dev_data: List[Instance],
    batch_size: int = 8,
    dev_batch_size: int = 64,
    batches_per_epoch: Optional[int] = None
) -> Tuple[DataLoader, DataLoader]:

    train_loader = SimpleDataLoader(
        train_data, batch_size, shuffle=True,
        batches_per_epoch=batches_per_epoch
    )
    dev_loader = SimpleDataLoader(
        dev_data, dev_batch_size,
        shuffle=False,
        #batches_per_epoch=batches_per_epoch
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
    lr: float = 0.001,
    optimizer: str = "AdamW",
    optimizer_params: Dict[str, Any] = None
) -> Trainer:
    parameters = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    opt = AdamWOptimizer
    if optimizer == "Adam":
        opt = AdamOptimizer
    elif optimizer == "AdaDelta":
        opt = AdadeltaOptimizer#  (parameters, lr=1.0, rho=0.9, eps=1e-6)
    optimizer = opt(parameters, lr=lr, **(optimizer_params or {}))  # type: ignore
    logging.info("Current Optimizer: %s " % optimizer)

    logging.info(f"Num epochs: {num_epochs}")

    trainer = GradientDescentTrainer(
        model=model,
        serialization_dir=serialization_dir,
        data_loader=train_loader,
        validation_metric="-loss" if model.IS_SIAMESE else "-loss",
        validation_data_loader=dev_loader,
        num_epochs=num_epochs,
        patience=patience,
        optimizer=optimizer,
        cuda_device=cuda_device,
        use_amp=True if cuda_device >= 0 else False
    )
    return trainer


def generate_all_data(
    token_features: Tuple[str, ...] = None,
    msd_features: Tuple[str, ...] = None,
    ratio_train: float = 1.0,
    batch_size: int = 16,
    dev_batch_size: int = 64,
    batches_per_epoch: Optional[int] = None,
    get_me_bert: GetMeBert = GetMeBert(),
    instance_type: str = "default",
    folder: str = "dataset/main",
    **tsv_reader_kwargs
) -> Tuple[DataLoader, DataLoader, Vocabulary, ClassificationTsvReader]:
    instance_type = instance_type.lower()
    if instance_type not in {"default", "siamese", "triplet"}:
        raise ValueError("`instance_type` must be one of "+str({"default", "siamese", "triplet"}))

    # Expects a siamese.txt file in the same directory as train.txt
    if instance_type in {"siamese", "triplet"}:
        # Samples are not siamese loaded. Siamese affects the other datasets
        siamese_reader = ClassificationTsvReader(
            token_features=token_features, msd_features=msd_features,
            get_me_bert=get_me_bert, instance_type="default",
            **tsv_reader_kwargs
        )
        siamese_samples = get_siamese_samples(siamese_reader, siamese_filepath=f"{folder}/siamese.txt")

        # Then we create the normal one
        dataset_reader = ClassificationTsvReader(
            token_features=token_features, msd_features=msd_features,
            get_me_bert=get_me_bert,
            instance_type=instance_type, siamese_samples=siamese_samples,
            token_indexers=siamese_reader.token_indexers,
            tokenizer=siamese_reader.tokenizer,
            **tsv_reader_kwargs
        )
        # And parse the original data
        train_data, dev_data = read_data(dataset_reader, use_siamese_set=False, folder=folder)
    else:
        dataset_reader = ClassificationTsvReader(
            token_features=token_features, msd_features=msd_features,
            get_me_bert=get_me_bert, instance_type=instance_type,
            **tsv_reader_kwargs
        )
        train_data, dev_data = read_data(dataset_reader, use_siamese_set=True, folder=folder)

    if ratio_train != 1.0:
        train_data = train_data[:math.ceil(ratio_train*len(train_data))]

    vocab = build_vocab(
        train_data,
        list(map(get_metadata_namespace, dataset_reader.metadata_tokens_categories)) \
            if dataset_reader.metadata_encoding == MetadataEncoding.AS_CATEGORICAL else [])

    train_loader, dev_loader = build_data_loaders(
        train_data, dev_data,
        batch_size=batch_size, batches_per_epoch=batches_per_epoch,
        dev_batch_size=dev_batch_size
    )
    train_loader.index_with(vocab)
    dev_loader.index_with(vocab)

    return train_loader, dev_loader, vocab, dataset_reader


def train_model(
    model: BaseModel,
    train_loader: DataLoader,
    dev_loader: DataLoader,
    cuda_device: Union[List[int], int] = -1,
    patience: int = 2,
    num_epochs: int = 5,
    lr: float = 1e-4,
    optimizer: str = "AdamW",
    optimizer_params: Dict[str, Any] = None
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
            patience=patience,
            lr=lr,
            optimizer=optimizer,
            optimizer_params=optimizer_params
        )
        logging.info("Starting training")
        trainer.train()
        logging.info("Finished training")
    return

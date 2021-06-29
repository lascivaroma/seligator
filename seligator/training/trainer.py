import tempfile
import math
from typing import Tuple, List, Iterable, Callable, Union, Optional
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


from seligator.dataset.tsv import build_dataset_reader


def read_data(reader: DatasetReader) -> Tuple[List[Instance], List[Instance]]:
    print("Reading data")
    training_data = list(reader.read("dataset/split/train.txt"))
    validation_data = list(reader.read("dataset/split/dev.txt"))
    return training_data, validation_data


def build_vocab(instances: Iterable[Instance]) -> Vocabulary:
    print("Building the vocabulary")
    return Vocabulary.from_instances(instances)


def build_data_loaders(
    train_data: List[Instance],
    dev_data: List[Instance],
    batch_size: int = 8,
    batches_per_epoch: Optional[int] = None
) -> Tuple[DataLoader, DataLoader]:
    train_loader = SimpleDataLoader(train_data, batch_size, shuffle=True, batches_per_epoch=batches_per_epoch)
    dev_loader = SimpleDataLoader(dev_data, batch_size, shuffle=False, batches_per_epoch=batches_per_epoch)
    return train_loader, dev_loader


def build_trainer(
    model: Model,
    serialization_dir: str,
    train_loader: DataLoader,
    dev_loader: DataLoader,
    cuda_device: Union[int, List[int]],
    patience: int = 2,
    num_epochs: int = 5
) -> Trainer:
    parameters = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    optimizer = AdamOptimizer(parameters)  # type: ignore
    print(f"Num epochs: {num_epochs}")
    trainer = GradientDescentTrainer(
        model=model,
        serialization_dir=serialization_dir,
        data_loader=train_loader,
        validation_metric="+fscore",
        validation_data_loader=dev_loader,
        num_epochs=num_epochs,
        patience=patience,
        optimizer=optimizer,
        cuda_device=cuda_device
    )
    return trainer


def run_training_loop(
    build_model: Callable[[Vocabulary, Tuple[str, ...]], Model],
    cuda_device: Union[List[int], int] = -1,
    use_only: Tuple[str, ...] = None,
    patience: int = 2,
    num_epochs: int = 5,
    ratio_train: float = 1.0,
    batch_size: int = 16,
    batches_per_epoch: Optional[int] = None,
    bert_dir: Optional[str] = None
):
    dataset_reader = build_dataset_reader(
        use_only=use_only,
        bert_dir=bert_dir
    )
    train_data, dev_data = read_data(dataset_reader)

    if ratio_train != 1.0:
        train_data = train_data[:math.ceil(ratio_train*len(train_data))]

    vocab = build_vocab(train_data)
    if "token_subword" in use_only:
        model = build_model(
            vocab=vocab, use_only=use_only,
            bert_dir=bert_dir, bert_tokenizer=dataset_reader.token_indexers["token_subword"].tokenizer)
    else:
        model = build_model(vocab=vocab, use_only=use_only)

    if cuda_device != -1:
        model = model.cuda()

    train_loader, dev_loader = build_data_loaders(
        train_data, dev_data,
        batch_size=batch_size, batches_per_epoch=batches_per_epoch
    )
    train_loader.index_with(vocab)
    dev_loader.index_with(vocab)

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
        print("Starting training")
        trainer.train()
        print("Finished training")

    return model, dataset_reader


if __name__ == "__main__":
    from seligator.models import baseline
    model, dataset_reader = run_training_loop(
        build_model=baseline.build_model,
        cuda_device=-1,
        use_only=("token", ),
        num_epochs=100,
        batch_size=16,
        patience=None
    )


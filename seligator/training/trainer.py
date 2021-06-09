import tempfile
from typing import Tuple, List, Iterable, Callable, Union
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
) -> Tuple[DataLoader, DataLoader]:
    train_loader = SimpleDataLoader(train_data, 8, shuffle=True)
    dev_loader = SimpleDataLoader(dev_data, 8, shuffle=False)
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
    trainer = GradientDescentTrainer(
        model=model,
        serialization_dir=serialization_dir,
        data_loader=train_loader,
        validation_data_loader=dev_loader,
        num_epochs=num_epochs,
        patience=patience,
        optimizer=optimizer,
        cuda_device=cuda_device
    )
    return trainer


def run_training_loop(
    build_model: Callable[[Vocabulary], Model],
    cuda_device: Union[List[int], int] = -1,
    use_only: Tuple[str, ...] = None,
    patience: int = 2,
    num_epochs: int = 5
):
    dataset_reader = build_dataset_reader(use_only=use_only)
    train_data, dev_data = read_data(dataset_reader)

    vocab = build_vocab(train_data + dev_data)
    model = build_model(vocab)

    if cuda_device != -1:
        model = model.cuda()

    train_loader, dev_loader = build_data_loaders(train_data, dev_data)
    train_loader.index_with(vocab)
    dev_loader.index_with(vocab)

    # You obviously won't want to create a temporary file for your training
    # results, but for execution in binder for this guide, we need to do this.
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
    model, dataset_reader = run_training_loop(build_model=baseline.build_model, cuda_device=-1)


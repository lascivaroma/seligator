from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import csv
from typing import Dict, List, Union, Optional
from numpy import ndarray

from allennlp.data.data_loaders import SimpleDataLoader
from allennlp.training.util import evaluate
from allennlp.data import (
    DatasetReader,
    Instance
)

from seligator.models.baseline import SimpleClassifier


def represent(instance: Instance, prediction: Dict[str, ndarray], labels: Dict[int, str]
              ) -> Dict[str, Union[str, float]]:
    pred = labels[prediction["probs"].argmax()]
    return {
        "sentence": " ".join([tok.text for tok in instance.fields["token"].tokens]),
        "label": instance.fields["label"].label,
        "prediction": pred,
        "ok": str(pred == instance.fields["label"].label),
        **{
            f"score-{labels[idx]}": element
            for idx, element in enumerate(prediction["probs"].tolist())
        }
    }


def simple_batcher(lst: List, n: int) -> List:
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def run_tests(test_file: str, dataset_reader: DatasetReader, model: SimpleClassifier,
              dump: Optional[str] = None):
    print("Evaluating")
    test_data = list(dataset_reader.read(test_file))
    data_loader = SimpleDataLoader(test_data, batch_size=8, shuffle=False)
    data_loader.index_with(model.vocab)

    results = evaluate(model, data_loader, cuda_device=0)
    print(results)

    print("Evaluating: Predicting")
    output_test = []
    for batch in simple_batcher(test_data, 8):
        output_test.extend(model.forward_on_instances(batch))
    preds, truths = [], []

    labels = model.labels

    if dump:
        f = open("preds.csv", "w")
        writer = csv.DictWriter(f, fieldnames=["sentence", "label", "prediction", "ok"] + [
            f"score-{labels[idx]}" for idx in labels
        ])
        writer.writeheader()

    for inst, result in zip(test_data, output_test):
        rep = represent(inst, result, labels=labels)
        preds.append(rep["prediction"])
        truths.append(rep["label"])
        if dump:
            writer.writerow(rep)

    labels_list = [
        labels[idx]
        for idx in range(len(labels))
    ]
    disp = ConfusionMatrixDisplay(confusion_matrix(preds, truths, labels=labels_list), display_labels=labels_list)
    disp.plot()
    disp.figure_.show()

    del test_data  # Avoid stuff remaining
    return disp


if __name__ == "__main__":
    from seligator.training.trainer import run_training_loop
    from seligator.models.baseline import build_model

    model, reader = run_training_loop(build_model=build_model, cuda_device=0)
    run_tests("dataset/split/test.txt", dataset_reader=reader, model=model)

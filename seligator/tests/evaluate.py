from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import csv
from typing import Dict, List, Union, Optional, Tuple
from numpy import ndarray

from allennlp.data.data_loaders import SimpleDataLoader
from allennlp.training.util import evaluate
from allennlp.data import (
    DatasetReader,
    Instance
)

from seligator.models.base import BaseModel


def get_unknown_and_sentence(instance: Instance, field: str, mode: Optional[str] = None):
    field = instance.fields[field]

    sentence = " ".join([tok.text for tok in field.tokens])
    if mode == "subword":
        return sentence.replace(" ", "").replace("_", " ")
    return sentence


def represent(instance: Instance, prediction: Dict[str, ndarray], labels: Dict[int, str]
              ) -> Dict[str, Union[str, float]]:
    pred = labels[prediction["probs"].argmax()]  # ToDo: does not work with current Siamese network

    prefix = ""
    if "left_label" in instance.fields:
        prefix = "left_"

    sentence = " ".join(instance.fields[prefix+"metadata"]["sentence"])

    return {
        "sentence": sentence,
        "label": instance.fields[prefix+"label"].label,
        "prediction": pred,
        "ok": str(pred == instance.fields[prefix+"label"].label),
        **{
            f"score-{labels[idx]}": element
            for idx, element in enumerate(prediction["probs"].tolist())
        },
        **{
            additional_output: prediction[additional_output]
            for additional_output in ("bert_projection", "attention", "doc-vectors")
            if additional_output in prediction
        }
        # ToDo: Display unknown values ?
    }


def simple_batcher(lst: List, n: int) -> List:
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def run_tests(test_file: str, dataset_reader: DatasetReader, model: BaseModel,
              dump: Optional[str] = None) -> Tuple[Dict[str, float], ConfusionMatrixDisplay]:
    print("Evaluating")
    test_data = list(dataset_reader.read(test_file))
    data_loader = SimpleDataLoader(test_data, batch_size=8, shuffle=False)
    data_loader.index_with(model.vocab)

    results = evaluate(model, data_loader, cuda_device=0)

    print("Evaluating: Predicting")
    output_test = []
    for batch in simple_batcher(test_data, 8):
        output_test.extend(model.forward_on_instances(batch))
    preds, truths = [], []

    labels = model.labels

    if dump:
        f = open(dump, "w")
        writer = csv.DictWriter(f, fieldnames=["sentence", "label", "prediction", "ok",
                                               "bert_projection", "attention", "doc-vectors"] + [
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
    return results, disp

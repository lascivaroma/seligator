import csv
from typing import Optional, Tuple, Dict

from allennlp.data import DatasetReader
from allennlp.data.data_loaders import SimpleDataLoader
from allennlp.training.util import evaluate
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from seligator.models.base import BaseModel
from seligator.prediction.utils import represent, simple_batcher


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
        other_headers = represent(test_data[0], output_test[0], label_vocabulary=labels)
        f = open(dump, "w")
        fields = ["sentence", "label", "prediction", "ok",
                                               "bert_projection", "attention", "doc-vectors"] + [
            f"score-{labels[idx]}" for idx in labels
        ]
        for key in other_headers:
            if key not in fields:
                fields.append(key)
        # fields  # ToDo : as missing fields from instance.fields[prefix+"metadata"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

    for inst, result in zip(test_data, output_test):
        rep = represent(inst, result, label_vocabulary=labels)
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

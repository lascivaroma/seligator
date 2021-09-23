from typing import Optional, Dict, Union, List

from allennlp.data import Instance
from numpy.core._multiarray_umath import ndarray


def get_unknown_and_sentence(instance: Instance, field: str, mode: Optional[str] = None):
    field = instance.fields[field]

    sentence = " ".join([tok.text for tok in field.tokens])
    if mode == "subword":
        return sentence.replace(" ", "").replace("_", " ")
    return sentence


def represent(
        instance: Instance,
        prediction: Dict[str, ndarray],
        label_vocabulary: Dict[int, str],
        is_gt: bool = True,
) -> Dict[str, Union[str, float]]:
    pred = label_vocabulary[prediction["probs"].argmax()]  # ToDo: does not work with current Siamese network

    prefix = ""
    if "left_label" in instance.fields:
        prefix = "left_"

    metadata = {}
    if prefix+"token_subword" in instance.fields:
        metadata["token_subword"] = " ".join([tok.text for tok in instance.fields[prefix+"token_subword"].tokens])

    metadata.update({
      "sentence": " ".join(instance.fields[prefix+"metadata"]["sentence"]),
        **{k: val for k, val in instance.fields[prefix+"metadata"].items() if k != "sentence"}
    })

    out = {
        **metadata,
        "prediction": pred,
        "score-prediction": prediction["probs"].max(),
        **{
            additional_output: prediction[additional_output]
            for additional_output in ("bert_projection", "attention", "doc-vectors", "probs")
            if additional_output in prediction
        }
    }
    if is_gt:
        out.update({
            "label": instance.fields[prefix+"label"].label,
            "ok": str(pred == instance.fields[prefix+"label"].label),
            **{
                f"score-{label_vocabulary[idx]}": element
                for idx, element in enumerate(prediction["probs"].tolist())
            }
        })
    return out


def simple_batcher(lst: List, n: int) -> List:
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

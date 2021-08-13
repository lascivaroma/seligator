from typing import Dict, Any


def add_bootstrap(html, css):
    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Prediction</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/css/bootstrap.min.css" rel="stylesheet"
        integrity="sha384-EVSTQN3/azprG1Anm3QDgpJLIm9Nao0Yz1ztcQTwFspd3yD65VohhpuuCOmLASjC" crossorigin="anonymous">
    <style type="text/css">
        {css}
    </style>
</head>
<body>
    <div class="container">
        {html}
    </div>
</body>
</html>"""


def _replace_words(sentence: str) -> str:
    return "▣"*len(sentence.split())


def predictions_to_html(
        predictions: Dict[str, Any],
        styles=None,
        show_class: bool = True,
        repl_word: bool = False
) -> str:
    styles = styles or {"positive": "color:red;"}
    start, end = "", ""
    sentences = []
    html_template = """<div class="row">
    <div class="col-2">{start_end}</div>
    <div class="col-10"><ul class="list-unstyled">{sentences}</ul></div>
</div>"""

    css = "\n".join([
        "." + cls + " {" + st + "}"
        for cls, st in styles.items()
    ])
    ordered_dict = {}

    for sentence in predictions:
        cur_start, cur_end = sentence["start"], sentence["end"]
        if cur_start != start:
            if sentences:
                key = " ".join((start, end))
                if start == end:
                    key = start
                ordered_dict[key] = sentences
                sentences = []
            start = cur_start
        end = cur_end

        sentences.append((sentence["prediction"], sentence["score-prediction"], sentence["sentence"]))
        # print(f'{sentence["prediction"]} -> {sentence["sentence"]}')

    if sentences:
        key = " ".join((start, cur_end))
        if start == cur_end:
            key = start
        ordered_dict[key] = sentences

    def _format_sentence(cls: str, sent: str, score: float) -> str:
        if repl_word:
            sent = _replace_words(sent)
        if show_class:
            return f"<li class=\"{cls}\"><em>{cls}</em><small>{score:.2f}</small> {sent}</li>"
        else:
            return f"<li class=\"{cls}\"><small>{score:.2f}</small> {sent}</li>"

    return add_bootstrap("\n".join([
        html_template.format(
            start_end=start_end,
            sentences=" ".join([
                _format_sentence(cls, sentence, probs)
                for cls, probs, sentence in sentences
            ])
        )
        for start_end, sentences in ordered_dict.items()
    ]), css=css)

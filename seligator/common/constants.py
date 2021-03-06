from typing import Dict
import os

BERT_DIR = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../../bert/latin_bert"
)
CATS = ("Case", "Numb", "Gend", "Mood", "Tense", "Voice", "Person", "Deg")

MSD_CAT_NAME = "[msd]"

EMBEDDING_DIMENSIONS: Dict[str, int] = {
    "token": 100,
    "lemma": 100,
    "token_char": 100,
    "token_char_encoded": 150,
    "lemma_char": 100,
    "lemma_char_encoded": 150,
    "pos": 64,
    MSD_CAT_NAME: 64,
    **{
        cat.lower(): 64
        for cat in CATS
    }
}

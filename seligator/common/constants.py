from typing import Dict
CATS = ("Case", "Numb", "Gend", "Mood", "Tense", "Voice", "Person", "Deg")

EMBEDDING_DIMENSIONS: Dict[str, int] = {
    "token": 100,
    "lemma": 100,
    "token_char": 100,
    "token_char_encoded": 150,
    "pos": 10,
    **{
        cat.lower(): 10
        for cat in CATS
    }
}

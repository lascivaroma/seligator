from typing import Dict
CATS = ("Case", "Numb", "Gend", "Mood", "Tense", "Voice", "Person", "Deg")

EMBEDDING_DIMENSIONS: Dict[str, int] = {
    "token": 100,
    "lemma": 100,
    "token_character": 100,
    "pos": 10,
    **{
        cat: 10
        for cat in CATS
    }
}

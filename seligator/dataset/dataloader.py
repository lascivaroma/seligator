import os.path


def get_vocabulary_from_pretrained_embedding(file: str):
    for idx, line in enumerate(open(os.path.expanduser(file))):
        if idx == 0 or not line.strip():
            continue
        yield line.split()[0]

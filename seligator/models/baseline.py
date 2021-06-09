from typing import Dict, List, Optional

import torch.nn.functional
from allennlp.data import Vocabulary, TextFieldTensors
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder
from allennlp.nn import util

from seligator.models.base import BaseModel
from seligator.common.constants import EMBEDDING_DIMENSIONS


class SimpleClassifier(BaseModel):
    USES = ("token", )
    def __init__(self,
                 vocab: Vocabulary,
                 embedder: TextFieldEmbedder,
                 encoder: Seq2VecEncoder,
                 **kwargs):
        super().__init__(vocab)

        self.embedder = embedder
        self.encoder = encoder
        self.classifier = torch.nn.Linear(encoder.get_output_dim(), self.num_labels)

    def forward(self,
                token: TextFieldTensors,
                label: Optional[torch.Tensor] = None,
                **tasks) -> Dict[str, torch.Tensor]:

        token = {
            cat: token[cat]
            for cat in self.USES
        }

        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.embedder(token)
        # Shape: (batch_size, num_tokens)
        mask = util.get_text_field_mask(token)
        # Shape: (batch_size, encoding_dim)
        encoded_text = self.encoder(embedded_text, mask)
        # Shape: (batch_size, num_labels)
        logits = self.classifier(encoded_text)
        # Shape: (batch_size, num_labels)
        probs = torch.nn.functional.softmax(logits)
        # Shape: (1,)
        output = {"probs": probs}
        if label is not None:
            self.accuracy(logits, label)
            self.measure(logits, label)
            # Shape: (1,)
            output['loss'] = torch.nn.functional.cross_entropy(logits, label)
        return output


def build_model(vocab: Vocabulary, emb_dims: Dict[str, int] = None) -> Model:
    emb_dims = emb_dims or EMBEDDING_DIMENSIONS

    print("Building the model")
    vocab_size = vocab.get_vocab_size("token")
    embedder = BasicTextFieldEmbedder(
        {"token": Embedding(embedding_dim=emb_dims["token"], num_embeddings=vocab_size)}
    )
    encoder = BagOfEmbeddingsEncoder(embedding_dim=emb_dims["token"])
    return SimpleClassifier(vocab, embedder, encoder)

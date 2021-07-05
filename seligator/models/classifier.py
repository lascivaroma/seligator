from typing import Dict, Tuple, Optional, Callable

import torch.nn.functional
from allennlp.data import Vocabulary, TextFieldTensors
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder
from seligator.modules.embedders.latinBert import LatinPretrainedTransformer
from allennlp.nn import util

from seligator.models.base import BaseModel


class FeatureEmbeddingClassifier(BaseModel):
    def __init__(self,
                 vocab: Vocabulary,
                 input_features: Tuple[str, ...],

                 # Normal input
                 use_features: bool = True,
                 features_embedder: Optional[TextFieldEmbedder] = None,
                 features_encoder: Optional[Seq2VecEncoder] = None,

                 # Bert Input
                 use_bert: bool = False,
                 bert_embedder: Optional[LatinPretrainedTransformer] = None,
                 bert_pooler: Optional[Seq2VecEncoder] = None,

                 mixer: str = "concat",

                 emb_dropout: float = 0.3,
                 **kwargs):
        super().__init__(vocab, input_features=input_features)

        if emb_dropout:
            self._emb_dropout = torch.nn.Dropout(emb_dropout)
        else:
            self._emb_dropout = None

        self.use_features: bool = use_features
        self.features_embedder = features_embedder
        self.features_encoder = features_encoder

        self.use_bert = use_bert
        self.bert_embedder: LatinPretrainedTransformer = bert_embedder
        self.bert_pooler: Seq2VecEncoder = bert_pooler

        out_dim = 0
        if use_features and use_bert:
            if mixer == "concat":
                self.mixer = lambda inp1, inp2: torch.cat([inp1, inp2], -1)
                out_dim = self.features_encoder.get_output_dim() + self.bert_pooler.get_output_dim()
            elif mixer.startswith("Linear:"):
                out_dim = int(mixer.split(":")[-1])
                self._mixer = torch.nn.Linear(
                    self.features_encoder.get_output_dim() + self.bert_pooler.get_output_dim(),
                    out_dim
                )
                self.mixer = lambda inp1, inp2: self._mixer(torch.cat([inp1, inp2], -1))
        elif use_features:
            out_dim = self.features_encoder.get_output_dim()
        elif use_bert:
            out_dim = self.bert_pooler.get_output_dim()
        else:
            raise ValueError("Neither Bert or Features are used")

        self.classifier = torch.nn.Linear(out_dim, self.num_labels)

    def forward(self,
                label: Optional[torch.Tensor] = None,
                **inputs) -> Dict[str, torch.Tensor]:

        if self.use_bert:
            subw_inpt = inputs["token_subword"]["token_subword"]
            subw_embedded = self.bert_embedder(subw_inpt["token_ids"], mask=subw_inpt["mask"])
            subw_encoded = self.bert_pooler(subw_embedded, mask=subw_inpt["mask"])

        if self.use_features:
            token = self._rebuild_input(inputs)

            # Shape: (batch_size, num_tokens, embedding_dim)
            embedded_text = self.features_embedder(token)
            if self._emb_dropout is not None:
                # Shape: (batch_size, num_tokens, embedding_dim)
                embedded_text = self._emb_dropout(embedded_text)
            # Shape: (batch_size, num_tokens)
            mask = util.get_text_field_mask(token)
            # Shape: (batch_size, encoding_dim)
            encoded_text = self.features_encoder(embedded_text, mask)

        if self.use_bert and self.use_features:
            encoded_text = self.mixer(encoded_text, subw_encoded)
        elif self.use_bert:
            encoded_text = subw_encoded

        # Shape: (batch_size, num_labels)
        logits = self.classifier(encoded_text)

        # Shape: (batch_size, num_labels)
        probs = torch.nn.functional.softmax(logits)

        # Shape: (1,)
        output = {"probs": probs}
        if label is not None:
            self._compute_metrics(logits, label, output)

        return output

    @classmethod
    def build_model(
            cls,
            vocabulary: Vocabulary,
            emb_dims: Dict[str, int] = None,
            input_features: Tuple[str, ...] = ("token",),
            features_encoder: Callable[[int], Seq2VecEncoder] = BagOfEmbeddingsEncoder,
            char_encoders: Dict[str, Seq2VecEncoder] = None,
            **kwargs
    ) -> Model:
        emb = cls.build_embeddings(
            vocabulary=vocabulary,
            input_features=input_features,
            emb_dims=emb_dims,
            char_encoders=char_encoders
        )
        return cls(
            vocab=vocabulary,
            input_features=input_features,
            features_embedder=emb,
            features_encoder=features_encoder(emb.get_output_dim()),
            **kwargs
        )


if __name__ == "__main__":
    from allennlp.modules.seq2vec_encoders import LstmSeq2VecEncoder
    from seligator.training.trainer import generate_all_data, train_model
    from allennlp.modules.seq2vec_encoders import LstmSeq2VecEncoder, BertPooler
    from seligator.common.constants import EMBEDDING_DIMENSIONS

    # For test, just change the input feature here
    INPUT_FEATURES = ("token", "lemma", "token_char", "token_subword")
    USE_BERT = True
    BERT_DIR = "./bert/latin_bert"

    train, dev, vocab, reader = generate_all_data(input_features=INPUT_FEATURES, bert_dir=BERT_DIR)
    bert, bert_pooler = None, None
    if USE_BERT:
        bert = LatinPretrainedTransformer(
            BERT_DIR,
            tokenizer=reader.token_indexers["token_subword"].tokenizer,
            train_parameters=False
        )
        bert_pooler = BertPooler(BERT_DIR)

    embedding_encoders = {
        cat: LstmSeq2VecEncoder(
            input_size=EMBEDDING_DIMENSIONS[cat],
            hidden_size=EMBEDDING_DIMENSIONS[f"{cat}_encoded"],
            num_layers=2,
            bidirectional=True,
            dropout=.3
        )
        for cat in {"token_char", "lemma_char"}
    }

    model = FeatureEmbeddingClassifier.build_model(
        vocabulary=vocab,
        emb_dims=EMBEDDING_DIMENSIONS,
        input_features=INPUT_FEATURES,
        features_encoder=lambda input_dim: LstmSeq2VecEncoder(
            input_size=input_dim,
            hidden_size=128,
            dropout=0.3,
            bidirectional=True,
            num_layers=2
        ),
        char_encoders=embedding_encoders,

        use_bert=USE_BERT,
        bert_embedder=bert,
        bert_pooler=bert_pooler
    )

    model.cuda()

    train_model(
        model=model,
        train_loader=train,
        dev_loader=dev,
        cuda_device=0
    )

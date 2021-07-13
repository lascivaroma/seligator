from typing import Dict, Tuple, Optional

from torch import Tensor, nn
import torch.nn.functional as F
from allennlp.data import Vocabulary

from seligator.models.base import BaseModel
from seligator.modules.mixed_encoders import MixedEmbeddingEncoder


class FeatureEmbeddingClassifier(BaseModel):
    def __init__(self,
                 vocab: Vocabulary,
                 input_features: Tuple[str, ...],
                 mixed_encoder: MixedEmbeddingEncoder,
                 **kwargs):
        super().__init__(vocab, input_features=input_features)

        self.mixed_encoder: MixedEmbeddingEncoder = mixed_encoder
        self.classifier = nn.Linear(self.mixed_encoder.get_output_dim(), self.num_labels)

    def forward(self,
                label: Optional[Tensor] = None,
                **mixed_features) -> Dict[str, Tensor]:
        encoded_text, additional_out = self.mixed_encoder(mixed_features)

        # Shape: (batch_size, num_labels)
        logits = self.classifier(encoded_text)

        # Shape: (batch_size, num_labels)
        probs = F.softmax(logits)

        # Shape: (1,)
        output = {"probs": probs, **(additional_out or {})}
        if label is not None:
            self._compute_metrics(logits, label, output)

        return output


if __name__ == "__main__":
    from allennlp.modules.seq2vec_encoders import LstmSeq2VecEncoder
    from seligator.training.trainer import generate_all_data, train_model
    from allennlp.modules.seq2vec_encoders import LstmSeq2VecEncoder, BertPooler
    from seligator.common.constants import EMBEDDING_DIMENSIONS
    from seligator.common.bert_utils import what_type_of_bert

    # For test, just change the input feature here
    INPUT_FEATURES = ("token", "lemma", "token_char", "token_subword")
    USE_BERT = True
    BERT_DIR = "./bert/latin_bert"

    if USE_BERT:
        get_me_bert = what_type_of_bert(BERT_DIR, trainable=False, hugginface=False)
    else:
        get_me_bert = what_type_of_bert()

    train, dev, vocab, reader = generate_all_data(input_features=INPUT_FEATURES, get_me_bert=get_me_bert)

    bert, bert_pooler = None, None
    if USE_BERT:
        bert = get_me_bert.embedder
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

    model = FeatureEmbeddingClassifier(
        vocab=vocab,
        input_features=INPUT_FEATURES,
        mixed_encoder=MixedEmbeddingEncoder.build(
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
    )

    model.cuda()

    train_model(
        model=model,
        train_loader=train,
        dev_loader=dev,
        cuda_device=0
    )

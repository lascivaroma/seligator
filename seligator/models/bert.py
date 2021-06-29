from allennlp.modules.token_embedders import TokenCharactersEncoder

from typing import Dict, Tuple, Optional

import torch.nn.functional
from allennlp.data import Vocabulary, TextFieldTensors
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2vec_encoders import LstmSeq2VecEncoder, BertPooler
from allennlp.nn import util

from seligator.models.base import BaseModel
from seligator.common.constants import EMBEDDING_DIMENSIONS
from seligator.modules.latinBert import LatinPretrainedTransformer
from seligator.dataset.tokenizer import SubwordTextEncoderTokenizer


class SimpleRNNClassifier(BaseModel):
    def __init__(self,
                 vocab: Vocabulary,
                 input_feature_names: Tuple[str, ...],
                 bert_encoder: LatinPretrainedTransformer,
                 bert_pooler: BertPooler,
                 embedder: Optional[TextFieldEmbedder] = None,
                 encoder: Optional[BertPooler] = None,
                 emb_dropout: float = 0.3,
                 **kwargs):
        super().__init__(vocab, input_feature_names=input_feature_names)

        self.bert_embedder = bert_encoder
        self.bert_pooler = bert_pooler

        self.embedder: Optional[TextFieldEmbedder] = embedder
        self.encoder: Optional[BertPooler] = encoder

        classifier_in_dim = bert_pooler.get_output_dim()

        # If we merge multiple information
        self.additional_input: bool = False

        if self.encoder is not None and self.embedder is not None:
            self.additional_input = True
            classifier_in_dim += encoder.get_output_dim()

            if emb_dropout:
                self._emb_dropout = torch.nn.Dropout(emb_dropout)
            else:
                self._emb_dropout = None

        self.classifier = torch.nn.Linear(classifier_in_dim, self.num_labels)

    def forward(self,
                label: Optional[torch.Tensor] = None,
                **inputs) -> Dict[str, torch.Tensor]:

        subw_inpt = inputs["token_subword"]
        subw_mask = util.get_text_field_mask(subw_inpt)

        embedded = self.bert_embedder(subw_inpt["token_subword"]["token_ids"], mask=subw_mask)
        encoded = self.bert_pooler(embedded, mask=subw_mask)

        if self.additional_input:
            token: Dict[str, Dict[str, torch.Tensor]] = {
                cat: inputs[cat][cat]
                for cat in self.input_feature_names
                if cat != "token_subwords"
            }

            # Shape: (batch_size, num_tokens, embedding_dim)
            embedded_text = self.embedder(text_field_input=token)

            if self._emb_dropout is not None:
                embedded_text = self._emb_dropout(embedded_text)

            # Shape: (batch_size, num_tokens)
            mask = util.get_text_field_mask(token)

            # Shape: (batch_size, encoding_dim)
            encoded_text = self.encoder(embedded_text, mask)

            encoded = torch.cat((encoded, encoded_text), dim=-1)

        # Shape: (batch_size, num_labels)
        logits = self.classifier(encoded)

        # Shape: (batch_size, num_labels)
        probs = torch.nn.functional.softmax(logits)
        # Shape: (1,)
        output = {"probs": probs}

        if label is not None:
            self._compute_metrics(logits, label, output)

        return output


def build_model(
        vocab: Vocabulary,
        emb_dims: Dict[str, int] = None,
        use_only: Tuple[str, ...] = ("token", "token_subword", ),
        bert_dir: str = None,
        bert_tokenizer: Optional[SubwordTextEncoderTokenizer] = None
) -> Model:
    emb_dims = emb_dims or EMBEDDING_DIMENSIONS

    if use_only != ("token_subword", ):
        raise Exception("Currently, it's impossible to use something else than token subword")

    embedder = BasicTextFieldEmbedder(
        {
            # Normal tokens
            **{
            cat: Embedding(embedding_dim=emb_dims[cat], num_embeddings=vocab.get_vocab_size(cat))
                for cat in use_only
                if not cat.endswith("_char") and not cat.endswith("_subword")
            },
            # Weirder tokens
            **{
                cat: TokenCharactersEncoder(
                    embedding=Embedding(
                        embedding_dim=emb_dims[cat],
                        num_embeddings=vocab.get_vocab_size(cat)
                    ),
                    encoder=LstmSeq2VecEncoder(
                        input_size=emb_dims[cat],
                        hidden_size=emb_dims[f"{cat}_encoded"],
                        num_layers=2,
                        bidirectional=True,
                        dropout=.3
                    ),
                    dropout=0.3
                )
                for cat in use_only
                if cat.endswith("_char")
            },
        }
    )
    bert = LatinPretrainedTransformer(bert_dir, tokenizer=bert_tokenizer)
    bert_pooler = BertPooler(bert_dir)

    return SimpleRNNClassifier(
        vocab=vocab,
        #classic_embedder=embedder,
        bert_encoder=bert,
        bert_pooler=bert_pooler,
        #encoder=encoder,
        input_feature_names=use_only
    )


if __name__ == "__main__":
    from seligator.training.trainer import run_training_loop
    import logging
    logging.getLogger().setLevel(logging.INFO)
    model, dataset_reader = run_training_loop(
        build_model=build_model,
        cuda_device=-1,
        bert_dir="bert/latin_bert",
        use_only=("token_subword", )
    )

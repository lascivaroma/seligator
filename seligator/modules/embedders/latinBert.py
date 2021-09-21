import logging
from overrides import overrides
from typing import Optional

from transformers import BertModel
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from allennlp.modules.scalar_mix import ScalarMix
from allennlp.modules.token_embedders import TokenEmbedder
from allennlp.modules.token_embedders.pretrained_transformer_embedder import PretrainedTransformerEmbedder

import torch

from seligator.dataset.tokenizer import LatinSubwordTextEncoderTokenizer

logger = logging.getLogger(__name__)


_model_cache = {}


def get(model_name: str, make_copy: bool, **kwargs,) -> BertModel:
    global _model_cache
    transformer = BertModel.from_pretrained(
        model_name,
        **kwargs,
    )

    _model_cache[model_name] = transformer

    if make_copy:
        import copy
        return copy.deepcopy(transformer)
    else:
        return transformer


@TokenEmbedder.register("latin_transformer")
class LatinPretrainedTransformer(PretrainedTransformerEmbedder):
    """
    Uses a pretrained model from `transformers` as a `TokenEmbedder`.

    Registered as a `TokenEmbedder` with name "pretrained_transformer".

    # Parameters

    model_name : `str`
        The name of the `transformers` model to use. Should be the same as the corresponding
        `PretrainedTransformerIndexer`.
    max_length : `int`, optional (default = `None`)
        If positive, folds input token IDs into multiple segments of this length, pass them
        through the transformer model independently, and concatenate the final representations.
        Should be set to the same value as the `max_length` option on the
        `PretrainedTransformerIndexer`.
    sub_module: `str`, optional (default = `None`)
        The name of a submodule of the transformer to be used as the embedder. Some transformers naturally act
        as embedders such as BERT. However, other models consist of encoder and decoder, in which case we just
        want to use the encoder.
    train_parameters: `bool`, optional (default = `True`)
        If this is `True`, the transformer weights get updated during training. If this is `False`, the
        transformer weights are not updated during training.
    eval_mode: `bool`, optional (default = `False`)
        If this is `True`, the model is always set to evaluation mode (e.g., the dropout is disabled and the
        batch normalization layer statistics are not updated). If this is `False`, such dropout and batch
        normalization layers are only set to evaluation mode when when the model is evaluating on development
        or test data.
    last_layer_only: `bool`, optional (default = `True`)
        When `True` (the default), only the final layer of the pretrained transformer is taken
        for the embeddings. But if set to `False`, a scalar mix of all of the layers
        is used.
    override_weights_file: `Optional[str]`, optional (default = `None`)
        If set, this specifies a file from which to load alternate weights that override the
        weights from huggingface. The file is expected to contain a PyTorch `state_dict`, created
        with `torch.save()`.
    override_weights_strip_prefix: `Optional[str]`, optional (default = `None`)
        If set, strip the given prefix from the state dict when loading it.
    load_weights: `bool`, optional (default = `True`)
        Whether to load the pretrained weights. If you're loading your model/predictor from an AllenNLP archive
        it usually makes sense to set this to `False` (via the `overrides` parameter)
        to avoid unnecessarily caching and loading the original pretrained weights,
        since the archive will already contain all of the weights needed.
    gradient_checkpointing: `bool`, optional (default = `None`)
        Enable or disable gradient checkpointing.
    tokenizer_kwargs: `Dict[str, Any]`, optional (default = `None`)
        Dictionary with
        [additional arguments](https://github.com/huggingface/transformers/blob/155c782a2ccd103cf63ad48a2becd7c76a7d2115/transformers/tokenization_utils.py#L691)
        for `AutoTokenizer.from_pretrained`.
    transformer_kwargs: `Dict[str, Any]`, optional (default = `None`)
        Dictionary with
        [additional arguments](https://github.com/huggingface/transformers/blob/155c782a2ccd103cf63ad48a2becd7c76a7d2115/transformers/modeling_utils.py#L253)
        for `AutoModel.from_pretrained`.
    """  # noqa: E501

    authorized_missing_keys = [r"position_ids$"]

    @overrides
    def __init__(
            self,
            model_name: str,
            tokenizer: LatinSubwordTextEncoderTokenizer,
            *,
            max_length: int = None,
            sub_module: str = None,
            train_parameters: bool = True,
            eval_mode: bool = False,
            use_layer: int = -1,
            merge_all_layers: bool = False,  # last_layer_only equivalent
            gradient_checkpointing: Optional[bool] = None,
            **kwargs
    ) -> None:
        super(TokenEmbedder, self).__init__()

        self.transformer_model: BertModel = get(model_name, True)

        if gradient_checkpointing is not None:
            self.transformer_model.config.update({"gradient_checkpointing": gradient_checkpointing})

        self.config = self.transformer_model.config
        if sub_module:
            assert hasattr(self.transformer_model, sub_module)
            self.transformer_model = getattr(self.transformer_model, sub_module)
        self._max_length = max_length

        # I'm not sure if this works for all models; open an issue on github if you find a case
        # where it doesn't work.
        self.output_dim = self.config.hidden_size

        self._scalar_mix: Optional[ScalarMix] = None
        if merge_all_layers:
            print("last_layer_only", merge_all_layers)
            self._scalar_mix = ScalarMix(self.config.num_hidden_layers)
            self.config.output_hidden_states = True

        self._layer_output: int = use_layer
        if self._layer_output > self.config.num_hidden_layers:
            raise ValueError(f"Asked for output {use_layer} but bigger "
                             f"than number of hidden layer {self.config.num_hidden_layers}")

        try:
            if self.transformer_model.get_input_embeddings().num_embeddings != tokenizer.vocab_size:
                self.transformer_model.resize_token_embeddings(tokenizer.vocab_size)
        except NotImplementedError:
            # Can't resize for transformers models that don't implement base_model.get_input_embeddings()
            logger.warning(
                "Could not resize the token embedding matrix of the transformer model. "
                "This model does not support resizing."
            )

        self._num_added_start_tokens = len(tokenizer.single_sequence_start_tokens)
        self._num_added_end_tokens = len(tokenizer.single_sequence_end_tokens)
        self._num_added_tokens = self._num_added_start_tokens + self._num_added_end_tokens

        self.train_parameters = train_parameters
        if not train_parameters:
            for param in self.transformer_model.parameters():
                param.requires_grad = False

        self.eval_mode = eval_mode
        if eval_mode:
            self.transformer_model.eval()

    @overrides
    def forward(
        self,
        token_ids: torch.LongTensor,
        mask: torch.BoolTensor,
        type_ids: Optional[torch.LongTensor] = None,
        segment_concat_mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:  # type: ignore
        """
        # Parameters

        token_ids: `torch.LongTensor`
            Shape: `[batch_size, num_wordpieces if max_length is None else num_segment_concat_wordpieces]`.
            num_segment_concat_wordpieces is num_wordpieces plus special tokens inserted in the
            middle, e.g. the length of: "[CLS] A B C [SEP] [CLS] D E F [SEP]" (see indexer logic).
        mask: `torch.BoolTensor`
            Shape: [batch_size, num_wordpieces].
        type_ids: `Optional[torch.LongTensor]`
            Shape: `[batch_size, num_wordpieces if max_length is None else num_segment_concat_wordpieces]`.
        segment_concat_mask: `Optional[torch.BoolTensor]`
            Shape: `[batch_size, num_segment_concat_wordpieces]`.

        # Returns

        `torch.Tensor`
            Shape: `[batch_size, num_wordpieces, embedding_size]`.

        """
        # Some of the huggingface transformers don't support type ids at all and crash when you supply
        # them. For others, you can supply a tensor of zeros, and if you don't, they act as if you did.
        # There is no practical difference to the caller, so here we pretend that one case is the same
        # as another case.
        if type_ids is not None:
            max_type_id = type_ids.max()
            if max_type_id == 0:
                type_ids = None
            else:
                if max_type_id >= self._number_of_token_type_embeddings():
                    raise ValueError("Found type ids too large for the chosen transformer model.")
                assert token_ids.shape == type_ids.shape

        fold_long_sequences = self._max_length is not None and token_ids.size(1) > self._max_length
        if fold_long_sequences:
            batch_size, num_segment_concat_wordpieces = token_ids.size()
            token_ids, segment_concat_mask, type_ids = self._fold_long_sequences(
                token_ids, segment_concat_mask, type_ids
            )

        transformer_mask = segment_concat_mask if self._max_length is not None else mask
        assert transformer_mask is not None
        # Shape: [batch_size, num_wordpieces, embedding_size],
        # or if self._max_length is not None:
        # [batch_size * num_segments, self._max_length, embedding_size]

        # We call this with kwargs because some of the huggingface models don't have the
        # token_type_ids parameter and fail even when it's given as None.
        # Also, as of transformers v2.5.1, they are taking FloatTensor masks.
        parameters = {
            "input_ids": token_ids,
            "attention_mask": transformer_mask.float(),
            "output_hidden_states": True
        }
        if type_ids is not None:
            parameters["token_type_ids"] = type_ids

        transformer_output: BaseModelOutputWithPoolingAndCrossAttentions = self.transformer_model(**parameters)
        if self._scalar_mix is not None:
            # The hidden states will also include the embedding layer, which we don't
            # include in the scalar mix. Hence the `[1:]` slicing.
            hidden_states = transformer_output.hidden_states[1:]
            embeddings = self._scalar_mix(hidden_states)
        elif self._layer_output != -1:
            # logger.warning(f"Outputing at layer {self._layer_output}")
            embeddings = transformer_output.hidden_states[self._layer_output]
        else:
            embeddings = transformer_output.last_hidden_state

        return embeddings

if __name__ == "__main__":
    get("bert/latin_bert/", False)

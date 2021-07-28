import torch

from allennlp.common.checks import ConfigurationError
from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder


class ModifiedPytorchSeq2VecWrapper(Seq2VecEncoder):
    """ This is a modified version of the `PytorchSeq2VecWrapper` class, where we returns the Module output instead
    of the last hidden values
    """

    def __init__(self, module: torch.nn.modules.RNNBase) -> None:
        # Seq2VecEncoders cannot be stateful.
        super().__init__(stateful=False)
        self._module = module
        try:
            if not self._module.batch_first:
                raise ConfigurationError("Our encoder semantics assumes batch is always first!")
        except AttributeError:
            pass

    def get_input_dim(self) -> int:
        return self._module.input_size

    def get_output_dim(self) -> int:
        try:
            is_bidirectional = self._module.bidirectional
        except AttributeError:
            is_bidirectional = False
        return self._module.hidden_size * (2 if is_bidirectional else 1)

    def forward(
        self, inputs: torch.Tensor, mask: torch.BoolTensor, hidden_state: torch.Tensor = None
    ) -> torch.Tensor:

        if mask is None:
            # In original code, note the case, but well...
            raise ValueError("There must be a mask ?")
        (
            module_output,  # (nb_words, 2 * hiddenSize)
            _,  # = States. We do not use the states
            restoration_indices,
        ) = self.sort_and_run_forward(self._module, inputs, mask, hidden_state)

        # Batch x SequenceLength x Hidden*BiDir*Layers
        module_output, _ = torch.nn.utils.rnn.pad_packed_sequence(
            # module_output:
            sequence=module_output,
            batch_first=self._module.batch_first
        )

        return module_output[restoration_indices]

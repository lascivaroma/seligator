from allennlp.modules.seq2vec_encoders import Seq2VecEncoder


class PassThroughModel(Seq2VecEncoder):
    with_attention: bool = False

    def __init__(self, input_dim: int):
        super(PassThroughModel, self).__init__()
        self._input_dim: int = input_dim
        self._output_dim: int = input_dim

    def get_input_dim(self) -> int:
        return self._input_dim

    def get_output_dim(self) -> int:
        return self._output_dim

    def forward(self, *args):
        return args

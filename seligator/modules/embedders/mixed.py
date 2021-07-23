import torch
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder

from seligator.common.constants import MSD_CAT_NAME


class FeatureAndTextEmbedder(torch.nn.Module):
    def __init__(
            self,
            text_embedder: BasicTextFieldEmbedder,
            feature_embedder_in: int,
            feature_embedder_out: int
    ):
        super(FeatureAndTextEmbedder, self).__init__()

        self.text_embedder = text_embedder

        self.feature_embedder = torch.nn.Linear(feature_embedder_in, feature_embedder_out)
        self._feature_embedder_out: int = feature_embedder_out

    def get_output_dim(self) -> int:
        return self.text_embedder.get_output_dim() + self._feature_embedder_out

    def forward(self, tokens):

        msd = self.feature_embedder(tokens[MSD_CAT_NAME].type(torch.float))
        txt = self.text_embedder({cat: val for cat, val in tokens.items() if cat != MSD_CAT_NAME})
        return torch.cat([txt, msd], dim=-1)

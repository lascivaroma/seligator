from typing import Tuple
from seligator.common.constants import MSD_CAT_NAME


def get_fields(token_features: Tuple[str, ...], msd_features: Tuple[str, ...], agglomerate_msd: bool)\
        -> Tuple[Tuple[str, ...], bool]:
    agglomerate_msd = agglomerate_msd and msd_features
    if agglomerate_msd:
        return (MSD_CAT_NAME, *token_features), True
    else:
        return (*token_features, *msd_features), False

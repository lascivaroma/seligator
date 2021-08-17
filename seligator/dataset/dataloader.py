import os.path
import math
import random
from collections import defaultdict
from typing import Sequence, Optional, Iterable, List, Iterator
from overrides import overrides

import torch

from allennlp.data import BatchSampler, Instance
from allennlp.data.data_loaders import SimpleDataLoader
from allennlp.data.samplers.batch_sampler import BatchSampler
from typing import Optional, List


from allennlp.common.util import lazy_groups_of
from allennlp.common.tqdm import Tqdm
from allennlp.data.data_loaders.data_loader import DataLoader, TensorDict
from allennlp.data.data_loaders.data_collator import DefaultDataCollator
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.vocabulary import Vocabulary
import allennlp.nn.util as nn_util


__all__ = ["get_vocabulary_from_pretrained_embedding", "AtLeastOneSampler", "SimpleSampledDataLoader"]


def get_vocabulary_from_pretrained_embedding(file: str):
    for idx, line in enumerate(open(os.path.expanduser(file))):
        if idx == 0 or not line.strip():
            continue
        yield line.split()[0]


class AtLeastOneSampler(BatchSampler):
    """
    precision-negative: 0.9598, precision-positive: 0.0826, recall-negative: 0.6678, recall-positive: 0.5169, fscore-negative: 0.7876, fscore-positive: 0.1424, sim-accuracy: 0.3423, accuracy: 0.6596, batch_loss: 1.1009, loss: 0.8281 ||: 100%
    1372/1373 [00:34<00:00, 39.38it/s]
    precision-negative: 0.8874, precision-positive: 0.0678, recall-negative: 0.4658, recall-positive: 0.3967, fscore-negative: 0.6109, fscore-positive: 0.1158, sim-accuracy: 0.5219, accuracy: 0.4596, batch_loss: 1.1171, loss: 1.1433 ||: 99%
    85/86 [00:01<00:00, 5
    """
    def __init__(self, batch_size: int, label_field: str = "label", drop_last: bool = True):
        self.drop_last = drop_last
        self.batch_size = batch_size
        self.label_field = label_field
        self._random = True

    def _get_classic_batches(self, instances: Sequence[Instance]) -> Iterable[List[int]]:
        idx = list(range(0, len(instances)))
        random.shuffle(idx)

        for batch_idx in range(self.get_num_batches(instances)):
            amount = self.get_batch_size()
            batch = []
            while idx and amount > 0:
                batch.append(idx.pop())
                amount -= 1
            if batch:
                yield batch

    def get_batch_indices(self, instances: Sequence[Instance]) -> Iterable[List[int]]:
        if instances[0].get(self.label_field) is None:
            yield from self._get_classic_batches(instances)

        labels = defaultdict(list)
        for idx, inst in enumerate(instances):
            labels[inst[self.label_field].human_readable_repr()].append(idx)

        # We retrieve which field is the smallest
        smallest_index, smallest_len = None, 0
        for key, val in labels.items():
            if len(val) < smallest_len or smallest_len == 0:
                smallest_index, smallest_len = key, len(val)
            random.shuffle(val)

        # We ensure that there is enough of batches
        num_batches = self.get_num_batches(instances)
        # ToDo: Actually, train twice
        if num_batches > smallest_len:
            # We can insert a little more than needed.
            subset = [
                random.choice(labels[smallest_index]) for _ in range(
                    random.randint(num_batches-smallest_len, num_batches+1)
                )
            ]
            labels[smallest_index].extend(subset)
            random.shuffle(labels[smallest_index])

        # Otherwise we start by dispatching in each batch
        batches = [
            [
                labels[smallest_index].pop(),
                *[labels[other_index].pop() for other_index in labels if other_index != smallest_index]
            ]
            for _ in range(self.get_num_batches(instances))
        ]
        remaining = labels[smallest_index]
        for index in labels:
            if index != smallest_index:
                remaining.extend(labels[index])

        random.shuffle(remaining)

        batch_size = self.get_batch_size()
        while batches:
            current_batch = batches.pop()
            cur_batch_size = batch_size - len(current_batch)
            while remaining and cur_batch_size > 0:
                current_batch.append(remaining.pop())
                cur_batch_size -= 1
            if current_batch:
                # Do not forget to shuffle the batch, otherwise first is always positive
                random.shuffle(current_batch)
                yield current_batch

    def get_num_batches(self, instances: Sequence[Instance]) -> int:
        batch_count_float = len(instances) / self.batch_size
        if self.drop_last:
            return math.floor(batch_count_float)
        else:
            return math.ceil(batch_count_float)

    def get_batch_size(self) -> Optional[int]:
        return self.batch_size


@DataLoader.register("simple-sampled", constructor="from_dataset_reader")
class SimpleSampledDataLoader(SimpleDataLoader):
    """
    A very simple `DataLoader` that is mostly used for testing.
    """

    def __init__(
        self,
        instances: List[Instance],
        batch_size: int,
        *,
        sampler: Optional[AtLeastOneSampler] = None,
        shuffle: bool = False,
        batches_per_epoch: Optional[int] = None,
        vocab: Optional[Vocabulary] = None,
    ) -> None:
        super(SimpleSampledDataLoader, self).__init__(
            instances=instances,
            batch_size=batch_size,
            shuffle=shuffle,
            batches_per_epoch=batches_per_epoch,
            vocab=vocab
        )
        self.sampler = sampler

    def _iter_batches(self) -> Iterator[TensorDict]:
        if not self.sampler:
            return super(SimpleSampledDataLoader, self)._iter_batches()

        for batch in self.sampler.get_batch_indices(self.instances):
            # Batch is a list of int right now
            batch: List[Instance] = [self.instances[i] for i in batch]
            tensor_dict = self.collate_fn(batch)
            if self.cuda_device is not None:
                tensor_dict = nn_util.move_to_device(tensor_dict, self.cuda_device)
            yield tensor_dict

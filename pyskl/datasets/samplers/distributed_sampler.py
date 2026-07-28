# Copyright (c) OpenMMLab. All rights reserved.
import math
import torch
from collections import defaultdict
from torch.utils.data import DistributedSampler as _DistributedSampler


class DistributedSampler(_DistributedSampler):
    """DistributedSampler inheriting from
    ``torch.utils.data.DistributedSampler``.

    In pytorch of lower versions, there is no ``shuffle`` argument. This child
    class will port one to DistributedSampler.
    """

    def __init__(self,
                 dataset,
                 num_replicas=None,
                 rank=None,
                 shuffle=True,
                 seed=0):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        # for the compatibility from PyTorch 1.3+
        self.seed = seed if seed is not None else 0

    def __iter__(self):
        # deterministically shuffle based on epoch
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch + self.seed)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples
        return iter(indices)


class ClassSpecificDistributedSampler(_DistributedSampler):
    """ClassSpecificDistributedSampler inheriting from 'torch.utils.data.DistributedSampler'.

    Samples are sampled with a class specific probability (class_prob). This sampler is only applicable to single class
    recognition dataset. This sampler is also compatible with RepeatDataset.
    """

    def __init__(self,
                 dataset,
                 num_replicas=None,
                 rank=None,
                 class_prob=None,
                 shuffle=True,
                 seed=0):

        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle
        if class_prob is not None:
            if isinstance(class_prob, list):
                class_prob = {i: n for i, n in enumerate(class_prob)}
            assert isinstance(class_prob, dict)
        self.class_prob = class_prob
        # for the compatibility from PyTorch 1.3+
        self.seed = seed if seed is not None else 0

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        class_prob = self.class_prob
        dataset_name = type(self.dataset).__name__
        dataset = self.dataset if dataset_name != 'RepeatDataset' else self.dataset.dataset
        times = 1
        if dataset_name == 'RepeatDataset':
            times = self.dataset.times
            class_prob = {k: v * times for k, v in class_prob.items()}

        labels = [x['label'] for x in dataset.video_infos]
        samples = defaultdict(list)
        for i, lb in enumerate(labels):
            samples[lb].append(i)

        indices = []
        for class_idx, class_indices in samples.items():
            mul = class_prob.get(class_idx, times)
            for i in range(int(mul // 1)):
                indices.extend(class_indices)
            rem = int((mul % 1) * len(class_indices))
            inds = torch.randperm(len(class_indices), generator=g).tolist()
            indices.extend([class_indices[inds[i]] for i in range(rem)])

        if self.shuffle:
            shuffle = torch.randperm(len(indices), generator=g).tolist()
            indices = [indices[i] for i in shuffle]

        # reset num_samples and total_size here.
        self.num_samples = math.ceil(len(indices) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples
        return iter(indices)


class ClassBalancedDistributedSampler(_DistributedSampler):
    """Distributed sampler with epoch-wise class-balanced index draws.

    For every epoch, this sampler draws ``epoch_size`` samples with replacement
    from a class distribution ``P(c) proportional to n_c ** class_sample_power``,
    where ``n_c`` is the number of pre-built dataset items in class ``c``. After
    a class is drawn, one item from that class is drawn uniformly. This is meant
    for datasets that already materialize candidate windows and need stochastic
    epoch sampling without forcing every redundant majority-class item to appear
    once per epoch.
    """

    def __init__(self,
                 dataset,
                 num_replicas=None,
                 rank=None,
                 class_sample_power=0.5,
                 epoch_size=None,
                 shuffle=True,
                 seed=0):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle
        self.seed = seed if seed is not None else 0
        self.class_sample_power = float(class_sample_power)
        if self.class_sample_power < 0:
            raise ValueError("class_sample_power must be non-negative")
        self.epoch_size = None if epoch_size is None else int(epoch_size)
        if self.epoch_size is not None and self.epoch_size <= 0:
            raise ValueError("epoch_size must be positive")
        self._reset_epoch_sizes()

    def _global_epoch_size(self):
        return self.epoch_size if self.epoch_size is not None else len(self.dataset)

    def _reset_epoch_sizes(self):
        epoch_size = self._global_epoch_size()
        self.num_samples = math.ceil(epoch_size / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def __len__(self):
        return self.num_samples

    def _base_dataset(self):
        dataset_name = type(self.dataset).__name__
        return self.dataset.dataset if dataset_name == 'RepeatDataset' else self.dataset

    def _class_indices(self):
        dataset = self._base_dataset()
        if not hasattr(dataset, 'video_infos'):
            raise AttributeError(
                "ClassBalancedDistributedSampler requires dataset.video_infos")

        samples = defaultdict(list)
        for idx, item in enumerate(dataset.video_infos):
            if 'label' not in item:
                raise KeyError("Class-balanced sampling requires item['label']")
            samples[int(item['label'])].append(idx)
        if not samples:
            raise ValueError("Cannot sample from an empty dataset")
        return {class_idx: samples[class_idx] for class_idx in sorted(samples)}

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        samples = self._class_indices()
        class_ids = list(samples.keys())
        counts = torch.tensor(
            [len(samples[class_idx]) for class_idx in class_ids],
            dtype=torch.double)
        weights = counts.pow(self.class_sample_power)
        if not torch.isfinite(weights).all() or float(weights.sum()) <= 0:
            raise ValueError("Invalid class sampling weights")
        probs = weights / weights.sum()

        epoch_size = self._global_epoch_size()
        class_draws = torch.multinomial(
            probs, epoch_size, replacement=True, generator=g).tolist()

        indices = []
        for draw in class_draws:
            class_idx = class_ids[draw]
            class_indices = samples[class_idx]
            pos = int(torch.randint(
                len(class_indices), (1,), generator=g).item())
            indices.append(class_indices[pos])

        if self.shuffle:
            order = torch.randperm(len(indices), generator=g).tolist()
            indices = [indices[i] for i in order]

        # Reset sizes here in case epoch_size was changed between epochs.
        self._reset_epoch_sizes()

        extra = self.total_size - len(indices)
        if extra > 0:
            repeats = math.ceil(extra / len(indices))
            indices += (indices * repeats)[:extra]
        assert len(indices) == self.total_size

        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples
        return iter(indices)

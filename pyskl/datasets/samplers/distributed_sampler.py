# Copyright (c) OpenMMLab. All rights reserved.
import math
import json
import os
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
    """Distributed sampler with epoch-wise weighted replacement sampling.

    For sample ``i`` from class ``y_i``, let ``n_yi`` be that class count. The
    sampler assigns item weight ``a_i = n_yi ** (class_sample_power - 1)`` and
    draws ``epoch_size`` indices with replacement every epoch. With
    ``class_sample_power=0.5``, this gives ``a_i = 1 / sqrt(n_yi)`` and expected
    class probability ``sqrt(n_c) / sum_k sqrt(n_k)``.
    """

    def __init__(self,
                 dataset,
                 num_replicas=None,
                 rank=None,
                 class_sample_power=0.5,
                 epoch_size=None,
                 shuffle=True,
                 seed=0,
                 sampler_indices_output_dir=None,
                 sampler_indices_output_prefix='sampled_indices'):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle
        self.seed = seed if seed is not None else 0
        self.class_sample_power = float(class_sample_power)
        if self.class_sample_power < 0:
            raise ValueError("class_sample_power must be non-negative")
        self.epoch_size = None if epoch_size is None else int(epoch_size)
        if self.epoch_size is not None and self.epoch_size <= 0:
            raise ValueError("epoch_size must be positive")
        self.sampler_indices_output_dir = sampler_indices_output_dir
        self.sampler_indices_output_prefix = sampler_indices_output_prefix
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

    def _labels_and_class_counts(self):
        dataset = self._base_dataset()
        if not hasattr(dataset, 'video_infos'):
            raise AttributeError(
                "ClassBalancedDistributedSampler requires dataset.video_infos")

        labels = []
        counts = defaultdict(int)
        for idx, item in enumerate(dataset.video_infos):
            if 'label' not in item:
                raise KeyError("Class-balanced sampling requires item['label']")
            label = int(item['label'])
            labels.append(label)
            counts[label] += 1
        if not labels:
            raise ValueError("Cannot sample from an empty dataset")
        return labels, {class_idx: counts[class_idx] for class_idx in sorted(counts)}

    def _save_epoch_indices(self, sampled_indices, padded_indices, labels,
                            class_counts):
        if self.rank != 0 or not self.sampler_indices_output_dir:
            return

        os.makedirs(self.sampler_indices_output_dir, exist_ok=True)
        path = os.path.join(
            self.sampler_indices_output_dir,
            f'{self.sampler_indices_output_prefix}_epoch_{self.epoch:04d}.json')
        class_counts_json = {
            str(class_idx): int(count)
            for class_idx, count in class_counts.items()
        }
        sampled_label_counts = defaultdict(int)
        for index in sampled_indices:
            sampled_label_counts[int(labels[index])] += 1
        payload = {
            'epoch': int(self.epoch),
            'seed': int(self.seed),
            'num_replicas': int(self.num_replicas),
            'class_sample_power': float(self.class_sample_power),
            'requested_epoch_size': int(self._global_epoch_size()),
            'natural_sampled_count': int(len(sampled_indices)),
            'ddp_total_size': int(len(padded_indices)),
            'ddp_padding_count': int(len(padded_indices) - len(sampled_indices)),
            'class_counts': class_counts_json,
            'sampled_label_counts': {
                str(class_idx): int(sampled_label_counts[class_idx])
                for class_idx in sorted(sampled_label_counts)
            },
            'sampled_indices': [int(index) for index in sampled_indices],
            'ddp_padded_indices': [int(index) for index in padded_indices],
        }
        tmp_path = path + '.tmp'
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, separators=(',', ':'))
            f.write('\n')
        os.replace(tmp_path, path)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        labels, class_counts = self._labels_and_class_counts()
        item_weights = torch.tensor(
            [
                float(class_counts[label]) ** (self.class_sample_power - 1.0)
                for label in labels
            ],
            dtype=torch.double)
        if not torch.isfinite(item_weights).all() or float(item_weights.sum()) <= 0:
            raise ValueError("Invalid class sampling weights")
        probs = item_weights / item_weights.sum()

        epoch_size = self._global_epoch_size()
        indices = torch.multinomial(
            probs, epoch_size, replacement=True, generator=g).tolist()

        if self.shuffle:
            order = torch.randperm(len(indices), generator=g).tolist()
            indices = [indices[i] for i in order]

        natural_indices = list(indices)

        # Reset sizes here in case epoch_size was changed between epochs.
        self._reset_epoch_sizes()

        extra = self.total_size - len(indices)
        if extra > 0:
            repeats = math.ceil(extra / len(indices))
            indices += (indices * repeats)[:extra]
        assert len(indices) == self.total_size

        self._save_epoch_indices(
            natural_indices, indices, labels=labels, class_counts=class_counts)

        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples
        return iter(indices)

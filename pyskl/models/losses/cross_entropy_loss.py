# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F

from ..builder import LOSSES
from .base import BaseWeightedLoss


def soft_cross_entropy(cls_score, target_probs, class_weight=None, tolerance=1e-4):
    """Cross entropy for probability-distribution targets."""
    assert cls_score.size() == target_probs.size(), (
        f'target_probs shape {tuple(target_probs.size())} must match '
        f'logits shape {tuple(cls_score.size())}')
    assert cls_score.dim() == 2, 'Only support 2-dim soft label'
    assert target_probs.dtype.is_floating_point, 'Soft labels must be floating point'
    assert torch.isfinite(target_probs).all(), 'Soft labels contain NaN or Inf'
    assert (target_probs >= 0).all(), 'Soft labels must be non-negative'
    sums = target_probs.sum(dim=1)
    assert torch.all(torch.abs(sums - 1) < tolerance), (
        'Each soft-label row must sum to one within tolerance')

    log_probs = F.log_softmax(cls_score, dim=1)
    if class_weight is not None:
        class_weight = class_weight.to(cls_score.device)
        log_probs = log_probs * class_weight.unsqueeze(0)

    per_sample = -(target_probs * log_probs).sum(dim=1)
    if class_weight is not None:
        return per_sample.sum() / torch.sum(class_weight.unsqueeze(0) * target_probs)
    return per_sample.mean()


@LOSSES.register_module()
class CrossEntropyLoss(BaseWeightedLoss):
    """Cross Entropy Loss.

    Support two kinds of labels and their corresponding loss type. It's worth
    mentioning that loss type will be detected by the shape of ``cls_score``
    and ``label``.
    1) Hard label: This label is an integer array and all of the elements are
        in the range [0, num_classes - 1]. This label's shape should be
        ``cls_score``'s shape with the `num_classes` dimension removed.
    2) Soft label(probablity distribution over classes): This label is a
        probability distribution and all of the elements are in the range
        [0, 1]. This label's shape must be the same as ``cls_score``. For now,
        only 2-dim soft label is supported.

    Args:
        loss_weight (float): Factor scalar multiplied on the loss.
            Default: 1.0.
        class_weight (list[float] | None): Loss weight for each class. If set
            as None, use the same weight 1 for all classes. Only applies
            to CrossEntropyLoss and BCELossWithLogits (should not be set when
            using other losses). Default: None.
    """

    def __init__(self, loss_weight=1.0, class_weight=None):
        super().__init__(loss_weight=loss_weight)
        self.class_weight = None
        if class_weight is not None:
            self.class_weight = torch.Tensor(class_weight)

    def _forward(self, cls_score, label, **kwargs):
        """Forward function.

        Args:
            cls_score (torch.Tensor): The class score.
            label (torch.Tensor): The ground truth label.
            kwargs: Any keyword argument to be used to calculate
                CrossEntropy loss.

        Returns:
            torch.Tensor: The returned CrossEntropy loss.
        """
        if cls_score.size() == label.size():
            # calculate loss for soft label

            assert len(kwargs) == 0, \
                ('For now, no extra args are supported for soft label, '
                 f'but get {kwargs}')
            loss_cls = soft_cross_entropy(
                cls_score,
                label,
                class_weight=self.class_weight,
            )
        else:
            # calculate loss for hard label

            if self.class_weight is not None:
                assert 'weight' not in kwargs, \
                    "The key 'weight' already exists."
                kwargs['weight'] = self.class_weight.to(cls_score.device)
            loss_cls = F.cross_entropy(cls_score, label, **kwargs)

        return loss_cls


@LOSSES.register_module()
class BCELossWithLogits(BaseWeightedLoss):
    """Binary Cross Entropy Loss with logits.

    Args:
        loss_weight (float): Factor scalar multiplied on the loss.
            Default: 1.0.
        class_weight (list[float] | None): Loss weight for each class. If set
            as None, use the same weight 1 for all classes. Only applies
            to CrossEntropyLoss and BCELossWithLogits (should not be set when
            using other losses). Default: None.
    """

    def __init__(self, loss_weight=1.0, class_weight=None):
        super().__init__(loss_weight=loss_weight)
        self.class_weight = None
        if class_weight is not None:
            self.class_weight = torch.Tensor(class_weight)

    def _forward(self, cls_score, label, **kwargs):
        """Forward function.

        Args:
            cls_score (torch.Tensor): The class score.
            label (torch.Tensor): The ground truth label.
            kwargs: Any keyword argument to be used to calculate
                bce loss with logits.

        Returns:
            torch.Tensor: The returned bce loss with logits.
        """
        if self.class_weight is not None:
            assert 'weight' not in kwargs, "The key 'weight' already exists."
            kwargs['weight'] = self.class_weight.to(cls_score.device)
        loss_cls = F.binary_cross_entropy_with_logits(cls_score, label,
                                                      **kwargs)
        return loss_cls

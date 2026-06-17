import logging
import warnings

from typing import List
from pytorch_lightning.utilities import rank_zero_only
import torch
import math
from torch import default_generator, randperm
from torch.utils.data.dataset import Subset
import numpy as np


def get_logger(name=__name__, level=logging.INFO) -> logging.Logger:
    """Initializes multi-GPU-friendly python logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in (
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "fatal",
        "critical",
    ):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


def random_split(dataset, lengths, generator=default_generator):
    """
    Randomly split a dataset into non-overlapping new datasets of given lengths.

    If a list of fractions that sum up to 1 is given,
    the lengths will be computed automatically as
    floor(frac * len(dataset)) for each fraction provided.

    After computing the lengths, if there are any remainders, 1 count will be
    distributed in round-robin fashion to the lengths
    until there are no remainders left.

    Optionally fix the generator for reproducible results, e.g.:

    >>> random_split(range(10), [3, 7], generator=torch.Generator().manual_seed(42))
    >>> random_split(range(30), [0.3, 0.3, 0.4], generator=torch.Generator(
    ...   ).manual_seed(42))

    Args:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths or fractions of splits to be produced
        generator (Generator): Generator used for the random permutation.
    """
    if math.isclose(sum(lengths), 1) and sum(lengths) <= 1:
        subset_lengths: List[int] = []
        for i, frac in enumerate(lengths):
            if frac < 0 or frac > 1:
                raise ValueError(f"Fraction at index {i} is not between 0 and 1")
            n_items_in_split = int(
                math.floor(len(dataset) * frac)  # type: ignore[arg-type]
            )
            subset_lengths.append(n_items_in_split)
        remainder = len(dataset) - sum(subset_lengths)  # type: ignore[arg-type]
        # add 1 to all the lengths in round-robin fashion until the remainder is 0
        for i in range(remainder):
            idx_to_add_at = i % len(subset_lengths)
            subset_lengths[idx_to_add_at] += 1
        lengths = subset_lengths
        for i, length in enumerate(lengths):
            if length == 0:
                warnings.warn(
                    f"Length of split at index {i} is 0. "
                    f"This might result in an empty dataset."
                )

    # Cannot verify that dataset is Sized
    if sum(lengths) != len(dataset):  # type: ignore[arg-type]
        raise ValueError(
            "Sum of input lengths does not equal the length of the input dataset!"
        )

    indices = randperm(sum(lengths), generator=generator).tolist()  # type: ignore[call-overload]
    return [
        Subset(dataset, indices[offset - length : offset])
        for offset, length in zip(_accumulate(lengths), lengths)
    ]


def _accumulate(iterable, fn=lambda x, y: x + y):
    """Return running totals.

    NOTE: This is a copy of a function that was removed from PyTorch 2.3,
    and was being used in random_split.
    Source: https://github.com/JaidedAI/EasyOCR/issues/1243#issuecomment-2078356445
    """
    # _accumulate([1,2,3,4,5]) --> 1 3 6 10 15
    # _accumulate([1,2,3,4,5], operator.mul) --> 1 2 6 24 120
    it = iter(iterable)
    try:
        total = next(it)
    except StopIteration:
        return
    yield total
    for element in it:
        total = fn(total, element)
        yield total


def slice_batch(tree, slice_size):
    from collections.abc import Mapping, Sequence
    """
    Recursively slice a nested structure so that all leaf sequences
    (tensors/arrays/lists/tuples) are truncated to [:slice_size] along
    their first dimension.

    Supports:
      - dict / Mapping
      - list / tuple
      - torch.Tensor
      - numpy.ndarray
    Everything else is returned as-is.
    """
    # Torch tensor
    if torch is not None and isinstance(tree, torch.Tensor):
        return tree[:slice_size]

    # Numpy array
    if np is not None and isinstance(tree, np.ndarray):
        return tree[:slice_size]

    # Dict-like: slice each value
    if isinstance(tree, Mapping):
        return {k: slice_batch(v, slice_size) for k, v in tree.items()}

    # Sequence (list/tuple), but *not* str/bytes
    if isinstance(tree, Sequence) and not isinstance(tree, (str, bytes, bytearray)):
        # First slice the sequence itself, then recurse on each element
        sliced_seq = tree[:slice_size]
        return type(tree)(slice_batch(v, slice_size) for v in sliced_seq)

    # Anything else: leave as-is
    return tree

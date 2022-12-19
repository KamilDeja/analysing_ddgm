import numpy as np
import torch

from random import shuffle

from torch.utils.data import Subset

from .wrapper import Subclass, AppendName, Permutation


def data_split(dataset, dataset_name, return_classes=False, return_task_as_class=False, num_batches=5, num_classes=10,
               random_split=False,
               limit_data=None, dirichlet_split_alpha=None, dirichlet_equal_split=True, reverse=False,
               limit_classes=-1, val_size = 0.3, seed=0):
    assert num_batches == 1

    rng = np.random.default_rng(seed=seed)
    res = []
    num_selected = int(val_size * len(dataset))
    train_set_indices_bitmask = torch.ones(len(dataset))
    validation_indices = rng.choice(range(len(dataset)), num_selected)
    train_set_indices_bitmask[validation_indices] = 0

    train_subset = Subset(dataset, torch.where(train_set_indices_bitmask == 1)[0])

    val_subset = Subset(dataset, torch.where(train_set_indices_bitmask == 0)[0])
    train_dataset_splits = {}
    val_dataset_splits = {}

    train_dataset_splits[0] = AppendName(train_subset, [0] * len(train_subset), return_classes=return_classes,
                                            return_task_as_class=return_task_as_class)
    val_dataset_splits[0] = AppendName(val_subset, [0] * len(train_subset), return_classes=return_classes,
                                          return_task_as_class=return_task_as_class)

    print(
        f"Prepared dataset with splits: {[(idx, len(data)) for idx, data in enumerate(train_dataset_splits.values())]}")
    print(
        f"Validation dataset with splits: {[(idx, len(data)) for idx, data in enumerate(val_dataset_splits.values())]}")

    return train_dataset_splits, val_dataset_splits, None

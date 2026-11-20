# Adapted from tinker-cookbook/utils/misc_utils.py at 1991897cba59ea9d9b0030d183d1867b28eba9d9.
# Modified for Search-R1: local synchronous text-search training; omit generic framework modes.
# SPDX-License-Identifier: Apache-2.0
# See third_party/tinker_cookbook/ for license and source mapping.
from collections.abc import Mapping, Sequence

import numpy as np


def safezip(*items):
    return zip(*items, strict=True)


def all_same(items):
    return bool(items) and all(item == items[0] for item in items)


def dict_mean(list_of_dicts: Sequence[Mapping[str, float | int | None]]) -> dict[str, float]:
    """Compute the element-wise mean across a list of dictionaries."""
    key2values: dict[str, list[float | int]] = {}
    for d in list_of_dicts:
        for k, v in d.items():
            if v is None:
                continue
            key2values.setdefault(k, []).append(v)
    return {k: float(np.mean(values)) for k, values in key2values.items() if values}


def split_list(items, count):
    if count <= 0 or count > len(items):
        raise ValueError("Invalid split count")
    edges = np.linspace(0, len(items), count + 1).astype(int)
    return [list(items[edges[i] : edges[i + 1]]) for i in range(count)]

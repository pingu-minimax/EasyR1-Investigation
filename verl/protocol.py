# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Implement base data transfer protocol between any two functions, modules.
We can subclass Protocol to define more detailed batch info with specific keys

Commit 098931530606d22f867fd121b1dcb3225a43661f: fix data proto
- Changed serialization method for performance optimization
- Updated non_blocking default behavior
- Modified batch processing logic
"""

import copy
import io
import pickle
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union

import numpy as np
import ray
import torch
from numpy.typing import NDArray
from tensordict import TensorDict
from torch.distributed import ProcessGroup
from torch.utils.data import DataLoader

__all__ = ["DataProto", "union_tensor_dict"]


def pad_dataproto_to_divisor(data: "DataProto", size_divisor: int) -> tuple["DataProto", int]:
    """Pad a DataProto to size divisible by size_divisor

    Args:
        data (DataProto): the unpadded DataProto
        size_divisor (int): size divisor

    Returns:
        data (DataProto): the padded DataProto
        pad_size (int)
    """
    assert isinstance(data, DataProto), "data must be a DataProto"
    if len(data) % size_divisor != 0:
        pad_size = size_divisor - len(data) % size_divisor
        padding_protos = []
        remaining_pad = pad_size
        while remaining_pad > 0:
            take_size = min(remaining_pad, len(data))
            padding_protos.append(data[:take_size])
            remaining_pad -= take_size

        data_padded = DataProto.concat([data] + padding_protos)
    else:
        pad_size = 0
        data_padded = data

    return data_padded, pad_size


def unpad_dataproto(data: "DataProto", pad_size: int) -> "DataProto":
    if pad_size != 0:
        data = data[:-pad_size]

    return data


def union_tensor_dict(tensor_dict1: TensorDict, tensor_dict2: TensorDict) -> TensorDict:
    """Union two tensordicts."""
    if tensor_dict1.batch_size != tensor_dict2.batch_size:
        raise ValueError(
            f"Two tensor dict must have identical batch size. Got {tensor_dict1.batch_size} and {tensor_dict2.batch_size}"
        )

    for key in tensor_dict2.keys():
        if key in tensor_dict1 and not torch.equal(tensor_dict1[key], tensor_dict2[key]):
            raise ValueError(f"Key already exists: {key}.")

        tensor_dict1[key] = tensor_dict2[key]

    return tensor_dict1


@dataclass
class DataProto:
    """
    A DataProto is a data structure that aims to provide a standard protocol for data exchange between functions.
    It contains a batch (TensorDict) and a meta_info (Dict). The batch is a TensorDict https://pytorch.org/tensordict/.
    TensorDict allows you to manipulate a dictionary of Tensors like a single Tensor. Ideally, the tensors with the
    same batch size should be put inside batch.
    """

    batch: Optional[TensorDict] = None
    non_tensor_batch: dict[str, NDArray] = field(default_factory=dict)
    meta_info: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.check_consistency()

    def __len__(self) -> int:
        if self.batch is not None:
            return self.batch.batch_size[0]
        elif self.non_tensor_batch is not None and len(self.non_tensor_batch) > 0:
            pivot_key = list(self.non_tensor_batch.keys())[0]
            return self.non_tensor_batch[pivot_key].shape[0]
        else:
            return 0

    def to(self, device: torch.device, non_blocking: bool = False) -> "DataProto":
        """Move the batch to device

        Args:
            device (torch.device): the device to move to.
            non_blocking (bool, optional): whether to use non-blocking mode. Defaults to False.

        Returns:
            DataProto: the current DataProto.

        NOTE: remember to use torch.cuda.synchronize() after self.to("cpu") to avoid weird number
        """
        if self.batch is not None:
            self.batch = self.batch.to(device, non_blocking=non_blocking)

        return self

    def check_consistency(self):
        """Check the consistency of the DataProto. Mainly for batch and non_tensor_batch"""
        if self.batch is not None:
            assert len(self.batch.batch_size) == 1, "only support num_batch_dims=1"

        if self.batch is not None and len(self.non_tensor_batch) != 0:
            assert len(self.batch.batch_size) == 1, "only support num_batch_dims=1 when non_tensor_batch is not empty."

            batch_size = self.batch.batch_size[0]
            for key, value in self.non_tensor_batch.items():
                assert len(value) == batch_size, f"key {key} length {len(value)} is not equal to bsz {batch_size}."

    @staticmethod
    def concat(data: list["DataProto"]) -> "DataProto":
        """Concat a list of DataProto. The batch is concatenated among dim=0."""
        batch_lst = [batch.batch for batch in data]
        new_batch = torch.cat(batch_lst, dim=0) if batch_lst[0] is not None else None
        return DataProto(batch=new_batch, meta_info=data[0].meta_info)

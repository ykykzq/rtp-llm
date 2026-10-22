"""Independent conversion of checkpoint tensors into local runtime-weight shards."""

from __future__ import annotations

import copy
import json
import math
import threading
from collections import Counter
from contextlib import nullcontext

import torch


def encode_runtime_key(layer_id: int | None, name: str) -> str:
    if layer_id is not None and (type(layer_id) is not int or layer_id < 0):
        raise ValueError("runtime layer id must be a non-negative integer or None")
    if not isinstance(name, str) or not name:
        raise ValueError("runtime weight name must be a nonempty string")
    return json.dumps([layer_id, name], separators=(",", ":"), ensure_ascii=True)


def decode_runtime_key(key: str) -> tuple[int | None, str]:
    try:
        value = json.loads(key)
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError("runtime key must contain [layer_id, name]")
        layer_id, name = value
        if encode_runtime_key(layer_id, name) != key:
            raise ValueError("runtime key must use its canonical JSON encoding")
        return layer_id, name
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid runtime weight key: {key!r}") from error


def validate_runtime_tensor(name, tensor, descriptor, device) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"runtime tensor {name!r} must be a torch.Tensor")
    if (
        list(tensor.shape) != descriptor["shape"]
        or str(tensor.dtype) != descriptor["dtype"]
    ):
        raise ValueError(
            f"runtime tensor layout mismatch for {name!r}: "
            f"shape={list(tensor.shape)} dtype={tensor.dtype}, expected={descriptor}"
        )
    if tensor.device != device:
        raise ValueError(
            f"runtime tensor {name!r} is on {tensor.device}, expected {device}"
        )
    if tensor.layout != torch.strided:
        raise ValueError(f"runtime tensor {name!r} must have strided layout")


class RuntimeSnapshotConverter:
    """One conversion round; never retains a reference to resident model tensors.

    The caller owns returned tensors and must retain them until outgoing DMA ends.
    Call ``close`` only after in-flight transfers finish. Inputs are copied on the
    configured device before they can outlive ``add``; CPU staging is never used.

    The byte budget covers owned pending inputs and outputs of the current call,
    plus a conservative receptor workspace estimate: four full source buffers at
    the widest source/compute/FP32 dtype and three final local output buffers.
    Plain identity AtomicWeight loads with standard TP splits and the identity
    device rewrite only reserve source dtype casts and three local output buffers.
    It is a preflight bound for normal fusion, casts, TP slicing and rewrites,
    not a CUDA allocator limit for arbitrary custom receptor kernels. Persistent
    snapshots already handed to the caller are accounted by the snapshot pool.
    """

    def __init__(
        self,
        *,
        receptor_index,
        load_config,
        runtime_layout,
        tensor_source_factory,
        device,
        max_pending_bytes: int,
        ignored_source_names=(),
    ) -> None:
        if type(max_pending_bytes) is not int or max_pending_bytes <= 0:
            raise ValueError("max_pending_bytes must be a positive integer")
        self._device = torch.device(device)
        self._max_pending_bytes = max_pending_bytes
        self._load_config = copy.copy(load_config)
        if getattr(load_config, "merge_lora", False):
            raise ValueError("runtime snapshots do not support load-time LoRA merging")
        if hasattr(load_config, "phy2log"):
            self._load_config.phy2log = copy.deepcopy(load_config.phy2log)
        self._receptors = tuple(
            (frozenset(needed), copy.copy(receptor), layer_id)
            for needed, receptor, layer_id in receptor_index
        )
        self._layout = {entry["name"]: copy.deepcopy(entry) for entry in runtime_layout}
        if len(self._layout) != len(runtime_layout):
            raise ValueError("duplicate runtime layout entry")
        self.required_names = frozenset(
            name for needed, _, _ in self._receptors for name in needed
        )
        self.ignored_source_names = frozenset(ignored_source_names) - self.required_names
        self._source_factory = tensor_source_factory
        self._consumers = Counter(
            name for needed, _, _ in self._receptors for name in needed
        )
        self._buffer = {}
        self._seen = set()
        self._emitted = set()
        self._pending = set(range(len(self._receptors)))
        self._finished = False
        self._closed = False
        self._failed = False
        self._lock = threading.Lock()
        self._stream = (
            torch.cuda.Stream(device=self._device)
            if self._device.type == "cuda"
            else None
        )

    @property
    def expected_runtime_layout(self) -> list[dict]:
        return [copy.deepcopy(self._layout[key]) for key in sorted(self._layout)]

    @property
    def pending_bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size() for tensor in self._buffer.values()
        )

    def _check_budget(self, required_bytes: int) -> None:
        if required_bytes > self._max_pending_bytes:
            raise MemoryError(
                f"runtime snapshot pending/workspace budget exceeded: "
                f"required_bytes={required_bytes} max_pending_bytes={self._max_pending_bytes}"
            )

    def _workspace_bytes(self, needed, receptor, layer_id) -> int:
        compute_dtype = (
            getattr(receptor, "data_type", None) or self._load_config.compute_dtype
        )
        # The empty scalar is metadata-only; no model payload is transferred to CPU.
        width = max(
            4, torch.empty((), dtype=compute_dtype, device="meta").element_size()
        )
        sources = sum(
            self._buffer[name].numel() * max(width, self._buffer[name].element_size())
            for name in needed
        )
        key = encode_runtime_key(layer_id, receptor.name)
        descriptor = self._layout[key]
        dtype = getattr(torch, descriptor["dtype"].removeprefix("torch."))
        output = (
            math.prod(descriptor["shape"])
            * torch.empty((), dtype=dtype, device="meta").element_size()
        )
        if self._is_plain_atomic(receptor):
            cast_width = torch.empty(
                (), dtype=compute_dtype, device="meta"
            ).element_size()
            casts = sum(
                self._buffer[name].numel() * cast_width
                for name in needed
                if self._buffer[name].dtype != compute_dtype
            )
            return casts + 3 * output
        return 4 * sources + 3 * output

    def _is_plain_atomic(self, receptor) -> bool:
        def identity(function):
            return (
                getattr(function, "__module__", None) == "rtp_llm.utils.model_weight"
                and getattr(function, "__name__", None) == "identity"
            )

        if (
            type(receptor).__module__ != "rtp_llm.model_loader.weight_module"
            or type(receptor).__name__ != "AtomicWeight"
            or len(receptor.weights) != 1
            or not identity(receptor.process_fun)
            or not identity(receptor.weights[0].merge_fun)
        ):
            return False
        rewrite = self._load_config.exported_device.maybe_rewrite_weight_by_key
        if (
            getattr(rewrite, "__module__", None) != "rtp_llm.device.device_base"
            or getattr(rewrite, "__qualname__", None)
            != "DeviceBase.maybe_rewrite_weight_by_key"
        ):
            return False
        split = receptor._get_split_func()
        return getattr(
            split, "__module__", None
        ) == "rtp_llm.utils.model_weight" and getattr(split, "__name__", None) in {
            "sp_id",
            "sp_0",
            "sp_neg1",
            "sp_0_pad8",
        }

    @torch.inference_mode()
    def add(
        self, named_tensors, is_last: bool = False
    ) -> list[tuple[str, torch.Tensor]]:
        with self._lock:
            if self._closed or self._finished or self._failed:
                raise RuntimeError(
                    "runtime snapshot converter is closed, finished or failed"
                )
            try:
                return self._add(named_tensors, is_last)
            except Exception:
                self._failed = True
                raise

    def _add(self, named_tensors, is_last):
        incoming = list(named_tensors)
        batch_names = set()
        incoming_bytes = 0
        for name, tensor in incoming:
            if name in self._seen or name in batch_names:
                raise KeyError(f"duplicate snapshot source tensor {name!r}")
            if name not in self.required_names:
                raise KeyError(f"unexpected snapshot source tensor {name!r}")
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"snapshot source {name!r} must be a torch.Tensor")
            if tensor.device != self._device or tensor.layout != torch.strided:
                raise ValueError(
                    f"snapshot source {name!r} must be strided on {self._device}"
                )
            batch_names.add(name)
            incoming_bytes += tensor.numel() * tensor.element_size()
        if is_last:
            missing = self.required_names - self._seen - batch_names
            if missing:
                raise KeyError(
                    f"incomplete snapshot sources: missing={sorted(missing)[:30]}"
                )
        self._check_budget(self.pending_bytes + incoming_bytes)
        outputs = []
        output_bytes = 0
        context = (
            torch.cuda.stream(self._stream)
            if self._stream is not None
            else nullcontext()
        )
        if self._stream is not None:
            self._stream.wait_stream(torch.cuda.current_stream(self._device))
        try:
            with context:
                for name, tensor in incoming:
                    self._buffer[name] = tensor.detach().clone(
                        memory_format=torch.contiguous_format
                    )
                self._seen.update(batch_names)
                source = self._source_factory(self._buffer)
                for index in sorted(self._pending):
                    needed, receptor, layer_id = self._receptors[index]
                    if not needed.issubset(self._buffer):
                        continue
                    self._check_budget(
                        self.pending_bytes
                        + output_bytes
                        + self._workspace_bytes(needed, receptor, layer_id)
                    )
                    loaded = receptor.load(
                        source, layer_id, str(self._device), self._load_config
                    )
                    for name, tensor in loaded.items():
                        key = encode_runtime_key(layer_id, name)
                        if key not in self._layout:
                            raise KeyError(
                                f"unexpected converted runtime tensor {key!r}"
                            )
                        if key in self._emitted:
                            raise KeyError(
                                f"duplicate converted runtime tensor {key!r}"
                            )
                        validate_runtime_tensor(
                            key, tensor, self._layout[key], self._device
                        )
                        data = tensor.detach().clone(
                            memory_format=torch.contiguous_format
                        )
                        outputs.append((key, data))
                        output_bytes += data.numel() * data.element_size()
                        self._emitted.add(key)
                    # A view returned by a receptor may pin a full source storage.
                    # Drop it before accounting the next receptor's workspace.
                    if loaded:
                        del tensor, data
                    self._pending.remove(index)
                    for name in needed:
                        self._consumers[name] -= 1
                        if self._consumers[name] == 0:
                            del self._buffer[name]
                    del loaded
        finally:
            # The sender may immediately reuse its IPC storage even on failure.
            if self._stream is not None:
                self._stream.synchronize()
        if is_last:
            missing = self._layout.keys() - self._emitted
            if missing or self._pending:
                raise KeyError(
                    f"incomplete runtime snapshot: missing={sorted(missing)[:30]}"
                )
            self._finished = True
        return outputs

    def close(self) -> None:
        with self._lock:
            if self._stream is not None:
                self._stream.synchronize()
            self._buffer.clear()
            self._pending.clear()
            self._consumers.clear()
            self._closed = True

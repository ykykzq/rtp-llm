from __future__ import annotations

import logging
import re
import threading
from typing import Any, Mapping, Sequence

import torch

from rtp_llm.model_loader.loader import ModelLoader
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.model_loader.tensor_source import TensorSource

# Assuming these imports are from your project and accessible
from rtp_llm.model_loader.weight_module import WeightModule

from .tipc import CudaIpcHelper, SharedMemIpcMeta, SharedMemoryIPCHelper

# Dictionary for renaming specific layer weight names from an external format
# (e.g., 'verl') to the internal 'rtp-llm' format.
RENAME_DICTIONARY = {
    # verl
    "embed_tokens.weight": "embedding",
    "norm.weight": "final_layernorm.gamma",
    "norm.bias": "final_layernorm.beta",
    "lm_head.weight": "lm_head",
    "input_layernorm.weight": "pre_layernorm_weights.gamma",
    "post_attention_layernorm.weight": "post_layernorm_weights.gamma",
    "self_attn.qkv_proj.weight": "self_attention_weights.query_weight.kernel",
    "self_attn.qkv_proj.bias": "self_attention_weights.query_weight.bias",
    "self_attn.o_proj.weight": "self_attention_weights.attention_output_weight.kernel",
    "mlp.gate_proj.weight": "ffn_weights.intermediate_weight.kernel",
    "mlp.up_proj.weight": "ffn_weights.intermediate_weight3.kernel",
    "mlp.down_proj.weight": "ffn_weights.intermediate_weight2.kernel",
    # roll - megatron
    "mbedding.word_embeddings.weight": "embedding",
    "self_attention.linear_proj.weight": "self_attention_weights.attention_output_weight.kernel",
    "self_attention.linear_proj.bias": "self_attention_weights.attention_output_weight.bias",
    "self_attention.linear_qkv.weight": "self_attention_weights.query_weight.kernel",
    "self_attention.linear_qkv.bias": "self_attention_weights.query_weight.bias",
    "mlp.linear_fc1.layer_norm_weight": "post_layernorm_weights.gamma",
    # ???
    "mlp.linear_fc1.weight": "",
}


def rename_function(layer_name: str) -> str:
    """
    Transforms a layer weight name from an external format (e.g., 'verl')
    into the format required by 'rtp-llm'.
    The input format is expected to be like 'model.layers.1.self_attn_qkv_proj.bias'.
    Args:
        layer_name: The layer weight name string from an external source.
    Returns:
        The transformed layer weight name in 'rtp-llm's internal format.
        For example, 'model.layers.1.self_attn_qkv_proj.bias' might become
        'self_attention_weights.query_weight.bias' if it matches a pattern
        and is in the RENAME_DICTIONARY.
    Error Handling:
        This function does not explicitly raise errors but performs string manipulations
        and dictionary lookups. If an unexpected `layer_name` format is provided,
        it might return a string that is not correctly transformed or recognized
        by downstream components.
    """
    # Remove the "model." prefix
    if layer_name.startswith("model."):
        name: str = layer_name[len("model.") :]
    elif layer_name.startswith("decoder."):
        name: str = layer_name[len("decoder.") :]
    else:
        name: str = layer_name
    if "layers" in layer_name:
        # Remove "layers." prefix
        name = name[len("layers.") :]
        # Remove the layer number and the dot following it (e.g., "1." from "1.self_attn...")
        # This assumes the format "layers.<number>.<rest_of_name>"
        first_dot_after_layers = name.find(".")
        if first_dot_after_layers != -1:
            name = name[first_dot_after_layers + 1 :]
        if name in RENAME_DICTIONARY:
            return RENAME_DICTIONARY[name]
        return name
    else:
        if name in RENAME_DICTIONARY:
            return RENAME_DICTIONARY[name]
        return name


class WeightManager:
    """
    Manages model weight updates, including renaming weights from an external
    source and handling inter-process communication (IPC) for tensor transfer.
    It ensures that incoming tensors are correctly processed and sharded/replicated
    as per the rtp-llm model's internal structure (e.g., for Tensor Parallelism (TP)
    or Pipeline Parallelism (PP)).
    """

    def __init__(
        self,
        device,
        weight: ModelWeights,
        model_weights_loader: ModelLoader,
        non_owned_global_weights: Sequence[str] = (),
    ) -> None:
        """
        Initializes the WeightManager with a model's weights, device information, and weight loader.
        """
        self._s_helper = SharedMemoryIPCHelper()

        # Use the explicit device/weights/loader passed in by the caller (e.g. BaseModel),
        # instead of relying on any global "engine" object.
        if isinstance(device, torch.device):
            self._device = device
        else:
            self._device = torch.device(device)

        self._weights: ModelWeights = weight
        self._weights_loader: ModelLoader = model_weights_loader
        self._weight_module = self._weights_loader._model_weights_info
        self._non_owned_global_weights = frozenset(non_owned_global_weights)
        self._working_stream: torch.cuda.Stream = torch.cuda.Stream(
            device=self._device,
        )
        # TODO: Consider the actual need for this lock. If updates are always
        # serialized via the server's request handling, a per-update lock might
        # be redundant or require finer-grained locking within _weights.update_...
        self._lock = threading.Lock()

    def extract_layer_number(self, s: str) -> int | None:
        """
        Extracts the layer number (an integer) from a string that follows
        the pattern 'layers.<number>'.
        Args:
            s: The input string, e.g., 'model.layers.2.mlp.gate_proj.weight'.
        Returns:
            The extracted layer number as an integer if found; otherwise, returns `None`.
        Error Handling:
            Returns `None` if the pattern 'layers.<number>' is not found,
            or if the captured group cannot be converted to an integer.
        """
        match = re.search(r"layers\.(\d+)", s)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                return None
        else:
            return None

    def update(self, req: dict[str, str]) -> None:
        """
        Receives an Inter-Process Communication (IPC) tensor description and
        updates the corresponding model weights.
        For models with Tensor Parallelism (TP) or Pipeline Parallelism (PP),
        this function expects the transmitted tensor to be a complete, unsharded tensor.
        It then handles the internal sharding or replication according to the
        rtp-llm's specific model parallelism configuration.
        Args:
            req: A dictionary containing the IPC request details. Expected keys are:
                 - "desc": A string describing the tensor's IPC metadata
                           (e.g., `CuIpcTensorMeta` or `SharedMemIpcMeta` encoded string).
                 - "name": A string representing the original name of the weight
                           (e.g., 'model.layers.1.self_attn_qkv_proj.bias').
                 - "method": A string indicating the IPC method used ("cuda_ipc" or "shm").
        Returns:
            None. The method updates internal model weights directly.
        Error Handling:
            - `KeyError`: If "desc", "name", or "method" fields are missing from `req`.
            - `ValueError`: If the "method" is invalid (not "cuda_ipc" or "shm"),
                            or if a layer weight name is invalid and its ID cannot be extracted.
            - `NotImplementedError`: If "cuda_ipc" method is attempted (currently disallowed).
            - `Exception`: If the tensor cannot be built from the IPC metadata (e.g., invalid descriptor).
                          This is a general catch-all for unexpected failures in `_t_helper.build_from_meta`.
        """
        if "desc" not in req:
            raise KeyError(
                "Update request is missing the 'desc' field. "
                "It must contain IPC tensor metadata."
            )
        if "name" not in req:
            raise KeyError(
                "Update request is missing the 'name' field. "
                "It must specify the weight name to update."
            )
        if "method" not in req:
            raise KeyError(
                "Update request is missing the 'method' field. "
                "It must specify the IPC method (e.g., 'cuda_ipc' or 'shm')."
            )
        method: str = req["method"]
        desc: str = req["desc"]
        name: str = req["name"]
        stored_name: str = name

        if method not in {"cuda_ipc", "shm"}:
            raise ValueError(
                f"Invalid IPC method '{method}' provided. Only 'cuda_ipc' and 'shm' are allowed."
            )
        tensor: torch.Tensor | None = None

        if method == "cuda_ipc":
            helper = CudaIpcHelper()
            tensor = helper.build_from_meta(bytes.fromhex(desc))
        else:  # method == "shm"
            sm_meta: SharedMemIpcMeta = SharedMemIpcMeta.decode(desc)
            tensor = self._s_helper.build_from_meta(sm_meta)

        if tensor is None:
            logging.error(
                f"Fail to build tensor from ipc description {desc}, method: {method}"
            )
            # This should ideally not be reached if build_from_meta consistently returns a tensor or raises an error.
            raise Exception(
                f"Failed to build tensor from IPC description '{desc}' using method '{method}'. Tensor is None."
            )

        self.update_from_tensor(stored_name, tensor)

    def update_from_tensor(self, name: str, tensor: torch.Tensor) -> None:
        """Apply an already-materialized tensor to the live model weights.

        Expects the complete, unsharded tensor; TP/PP sharding is applied here
        according to rtp-llm's own parallelism configuration.
        """
        stored_name: str = name

        logging.info(
            f"update weight request: {name}, shape: {tensor.shape}, device: {tensor.device}, dtype: {tensor.dtype}"
        )
        with torch.cuda.stream(self._working_stream):
            config = self._weights_loader.get_load_config()
            if "layers" in name:
                # This is a layer-specific weight
                layer_id: int | None = self.extract_layer_number(name)
                if layer_id is None:
                    raise ValueError(
                        f"Invalid layer weight name format: '{name}'. "
                        "Could not extract layer number. Expected format like 'model.layers.<id>...'"
                    )
                name: str = rename_function(name)
                fail: bool = True

                for receptor in self._weight_module.layer_weights[layer_id]:
                    if receptor.name == name or (
                        "ffn_weights" in name and receptor.name == "__ffn_weights__"
                    ):
                        assert isinstance(receptor, WeightModule)

                        # split tensor into shards
                        shard = receptor.update(
                            tensor=tensor,
                            device=self._device,
                            load_config=config,
                            module_name=name,
                        )
                        if isinstance(shard, dict):
                            shard = next(iter(shard.values()))

                        # update tensor weight
                        self._weights.update_layer_weight(
                            layer_id=layer_id, name=name, data=shard
                        )
                        fail = False

                if fail:
                    raise KeyError(
                        f"{stored_name} not found. wanted name list is {[w.name for w in self._weight_module.layer_weights[layer_id]]}"
                    )

            else:
                # weight is global weight

                name: str = rename_function(name)
                if name in self._non_owned_global_weights:
                    raise PermissionError(
                        f"global weight {name!r} is a non-owning alias; update its owner instead"
                    )
                fail: bool = True
                for weight in self._weight_module.weights:
                    if weight.name == name:
                        shard: dict = weight.update(
                            tensor,
                            self._device,
                            load_config=self._weights_loader.get_load_config(),
                        )
                        if isinstance(shard, dict):
                            shard = next(iter(shard.values()))
                        self._weights.update_global_weight(name=name, data=shard)
                        fail = False

                if fail:
                    raise KeyError(
                        f"{stored_name} not found. wanted name list is {[w.name for w in self._weight_module.weights]}"
                    )

            self._working_stream.synchronize()

    def _build_receptor_index(self):
        """Index every leaf receptor by the set of checkpoint tensor names it consumes.

        Used by update_from_hf_tensors so a receptor fires exactly when all of its HF
        source tensors have arrived (e.g. in_proj_qkv + in_proj_z -> in_proj_qkvz).

        Composites are expanded to their leaves: indexing one as a unit would make it wait
        for every leaf's source at once, so a single name the caller never sends would
        strand the whole group.
        """
        load_config = self._weights_loader.get_load_config()
        index = []
        for entry in self._weight_module.weights:
            for receptor in entry.get_components():
                index.append(
                    (
                        frozenset(receptor.get_tensor_names(None, load_config)),
                        receptor,
                        None,
                    )
                )
        for layer_id, entries in enumerate(self._weight_module.layer_weights):
            for entry in entries:
                for receptor in entry.get_components():
                    index.append(
                        (
                            frozenset(receptor.get_tensor_names(layer_id, load_config)),
                            receptor,
                            layer_id,
                        )
                    )
        logging.info(
            f"receptor index: {len(index)} leaf receptors over "
            f"{len(self._weight_module.layer_weights)} layers"
        )
        return index

    def update_from_hf_tensors(self, named_tensors, is_last: bool = False) -> None:
        """Apply trained weights streamed under their original HuggingFace names.

        Unlike update_from_tensor (one internal tensor per call), this path lets rtp-llm's
        own receptor machinery fuse/reorder multiple source tensors (qkvz, ba, ffn gate/up),
        which is required for Qwen3.5's linear-attention layers. Sources are buffered until a
        receptor's full source set is present, then consumed.

        The incoming tensors are only borrowed for the duration of the call, so anything
        still awaiting its fusion partners is copied before returning. Pass is_last on the
        final batch of a round: no partner can arrive after it, so a non-empty buffer then
        means those copies would stay resident for the life of the process.
        """
        with self._lock:
            if getattr(self, "_receptor_index", None) is None:
                self._receptor_index = self._build_receptor_index()
            if not hasattr(self, "_hf_buffer"):
                self._hf_buffer = {}

            for name, tensor in named_tensors:
                self._hf_buffer[name] = tensor
                self._hf_round_total = getattr(self, "_hf_round_total", 0) + 1

            load_config = self._weights_loader.get_load_config()
            device = str(self._device)
            source = _DictTensorSource(self._hf_buffer)

            with torch.cuda.stream(self._working_stream):
                progress = True
                while progress:
                    progress = False
                    for needed, receptor, layer_id in self._receptor_index:
                        if not needed or not needed.issubset(self._hf_buffer.keys()):
                            continue
                        loaded = receptor.load(source, layer_id, device, load_config)
                        for wname, data in loaded.items():
                            if layer_id is None:
                                if wname in self._non_owned_global_weights:
                                    continue
                                self._weights.update_global_weight(
                                    name=wname, data=data
                                )
                            else:
                                self._weights.update_layer_weight(
                                    layer_id=layer_id, name=wname, data=data
                                )
                        for src_name in needed:
                            self._hf_buffer.pop(src_name, None)
                        progress = True
                self._working_stream.synchronize()

            if is_last:
                round_total = getattr(self, "_hf_round_total", 0)
                self._hf_round_total = 0
                if self._hf_buffer:
                    stranded = sorted(self._hf_buffer)
                    self._hf_buffer.clear()
                    for probe in stranded[:3]:
                        referencing = [
                            sorted(needed)
                            for needed, _, _ in self._receptor_index
                            if probe in needed
                        ]
                        logging.warning(
                            f"stranded {probe!r} referenced by {len(referencing)} entry(ies): {referencing[:2]}"
                        )
                    detail = (
                        f"{len(stranded)} of {round_total} streamed weight(s) matched no receptor "
                        f"among {len(self._receptor_index)} indexed: {stranded[:30]}"
                    )
                    if round_total and len(stranded) * 4 > round_total:
                        raise KeyError(f"weight name mapping is broken: {detail}")
                    logging.warning(f"dropping {detail}")

            for name, tensor in self._hf_buffer.items():
                self._hf_buffer[name] = tensor.clone()


class _DictTensorSource(TensorSource):
    """In-memory TensorSource over a {checkpoint_name: tensor} dict for RL weight updates."""

    def __init__(self, buffer: Mapping[str, torch.Tensor]) -> None:
        self._buffer = buffer

    def load_tensor(self, name: str, data_type: Any = torch.float16):
        tensor = self._buffer[name]
        if tensor.is_floating_point() and data_type is not None:
            tensor = tensor.to(data_type)
        return [tensor]

    def has_tensor(self, name: str) -> bool:
        return name in self._buffer

    def get_database(self):
        return None

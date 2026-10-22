"""Dependency-isolated CPU checks of the real snapshot and receptor Python code.

This harness does not import native RTP operators or claim CUDA integration coverage.
"""

import ast
import copy
import functools
import importlib.util
import json
import logging
import threading
import traceback
import unittest
import weakref
from abc import ABC, abstractmethod
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]
LOADER = ROOT / "model_loader"


def load_definitions(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if getattr(node, "name", None) in names]
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[future, *nodes], type_ignores=[])
    )
    exec(compile(module, str(path), "exec"), namespace)  # noqa: S102 - repository definitions only


def runtime_module():
    path = LOADER / "runtime_snapshot.py"
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location("runtime_snapshot_cpu_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SNAPSHOT = runtime_module()
NS = {
    "torch": torch,
    "threading": threading,
    "Counter": Counter,
    "logging": logging,
    "TensorSource": object,
    "copy": copy,
    "json": json,
}
if SNAPSHOT is not None:
    NS.update(vars(SNAPSHOT))
load_definitions(
    LOADER / "weight_manager.py", {"WeightManager", "_DictTensorSource"}, NS
)
load_definitions(LOADER / "model_weight_info.py", {"ModelWeights"}, NS)
WeightManager = NS["WeightManager"]
ModelWeights = NS["ModelWeights"]


class Stream:
    def synchronize(self):
        pass


class Receptor:
    def __init__(self, name, sources, transform=None):
        self.name = name
        self.sources = frozenset(sources)
        self.transform = transform or (lambda values: torch.cat(values))

    def get_components(self):
        return [self]

    def get_tensor_names(self, layer_id, config):
        return self.sources

    def load(self, source, layer_id, device, config):
        values = [source.load_tensor(name, None)[0] for name in sorted(self.sources)]
        return {self.name: self.transform(values)}


def manager_for(receptors, resident, *, rank=0, nonowned=(), layer_receptors=()):
    manager = WeightManager.__new__(WeightManager)
    manager._lock = threading.Lock()
    manager._device = torch.device("cpu")
    manager._working_stream = Stream()
    manager._non_owned_global_weights = frozenset(nonowned)
    manager._weights = ModelWeights(len(layer_receptors), "cpu", torch.float32)
    manager._weights.global_weights.update(resident)
    manager._weight_module = SimpleNamespace(
        weights=receptors, layer_weights=layer_receptors
    )
    config = SimpleNamespace(
        tp_size=4,
        tp_rank=rank,
        lm_head_tp_size=4,
        lm_head_tp_rank=rank,
        ep_size=1,
        ep_rank=0,
        dp_size=1,
        dp_rank=0,
        ffn_tp_size=4,
        ffn_tp_rank=rank,
        hidden_size=2,
        head_num=4,
        head_num_kv=4,
        size_per_head=1,
        moe_pure_tp_mode=False,
        bit=16,
        merge_lora=False,
        compute_dtype=torch.float32,
        use_swizzleA=False,
        exported_device=SimpleNamespace(
            maybe_rewrite_weight_by_key=lambda name, tensor, **kw: tensor
        ),
    )
    manager._weights_loader = SimpleNamespace(get_load_config=lambda: config)
    return manager


class RuntimeSnapshotTest(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(SNAPSHOT, "runtime snapshot implementation is missing")
        self.stream_patch = patch.object(
            torch.cuda, "stream", return_value=nullcontext()
        )
        self.stream_patch.start()
        self.addCleanup(self.stream_patch.stop)

    def test_shared_dependencies_are_owned_and_never_change_resident_weights(self):
        joint = Receptor("joint", {"shared", "other"})
        single = Receptor("single", {"shared"}, lambda ts: ts[0])
        manager = manager_for(
            [joint, single], {"joint": torch.zeros(3), "single": torch.zeros(2)}
        )
        manager._hf_buffer = {"unrelated": torch.tensor([99.0])}
        converter = manager.create_runtime_snapshot_converter(4096)
        self.assertEqual(converter.required_names, {"shared", "other"})
        incoming = torch.tensor([1.0, 2.0])
        first = dict(converter.add([("shared", incoming)]))
        self.assertEqual(first['[null,"single"]'].tolist(), [1.0, 2.0])
        self.assertNotEqual(first['[null,"single"]'].data_ptr(), incoming.data_ptr())
        incoming.fill_(77)
        second = dict(converter.add([("other", torch.tensor([3.0]))], is_last=True))
        self.assertEqual(second['[null,"joint"]'].tolist(), [3.0, 1.0, 2.0])
        self.assertEqual(manager._weights.global_weights["joint"].tolist(), [0, 0, 0])
        self.assertEqual(manager._weights.global_weights["single"].tolist(), [0, 0])
        self.assertEqual(set(manager._hf_buffer), {"unrelated"})
        converter.close()
        self.assertEqual(first['[null,"single"]'].tolist(), [1, 2])

    def test_output_is_contiguous_with_owning_storage(self):
        receptor = Receptor("out", {"source"}, lambda ts: ts[0].T)
        manager = manager_for([receptor], {"out": torch.zeros(3, 2)})
        source = torch.arange(6.0).reshape(2, 3)
        converter = manager.create_runtime_snapshot_converter(4096)
        output = converter.add([("source", source)], is_last=True)[0][1]
        self.assertTrue(output.is_contiguous())
        self.assertIsNone(output._base)
        source.zero_()
        self.assertEqual(output.tolist(), [[0, 3], [1, 4], [2, 5]])

    def test_nonowned_alias_and_source_free_constant_are_excluded(self):
        manager = manager_for(
            [
                Receptor("embedding", {"embed"}),
                Receptor("alias", {"alias_source"}),
                Receptor("constant", set()),
            ],
            {
                "embedding": torch.zeros(2),
                "alias": torch.zeros(2),
                "constant": torch.ones(2),
            },
            nonowned={"alias"},
        )
        converter = manager.create_runtime_snapshot_converter(4096)
        self.assertEqual(converter.required_names, {"embed"})
        self.assertEqual(converter.ignored_source_names, {"alias_source"})
        self.assertEqual(
            manager.get_runtime_weight_layout(),
            [
                {"name": '[null,"embedding"]', "shape": [2], "dtype": "torch.float32"},
            ],
        )

    def test_nonowned_alias_does_not_ignore_shared_required_source(self):
        manager = manager_for(
            [Receptor("embedding", {"shared"}), Receptor("alias", {"shared"})],
            {"embedding": torch.zeros(2), "alias": torch.zeros(2)},
            nonowned={"alias"},
        )
        converter = manager.create_runtime_snapshot_converter(4096)
        self.assertEqual(converter.required_names, {"shared"})
        self.assertEqual(converter.ignored_source_names, set())
        with self.assertRaisesRegex(KeyError, "unexpected"):
            converter.add([("alias_source", torch.ones(2))])

    def test_rejects_duplicate_missing_and_unexpected_sources(self):
        for batches, message in [
            (
                [([("left", torch.ones(1)), ("left", torch.ones(1))], False)],
                "duplicate",
            ),
            (
                [
                    ([("left", torch.ones(1))], False),
                    ([("left", torch.ones(1))], False),
                ],
                "duplicate",
            ),
            ([([("left", torch.ones(1))], True)], "missing"),
            ([([("vision.weight", torch.ones(1))], False)], "unexpected"),
        ]:
            with self.subTest(message=message, batches=len(batches)):
                manager = manager_for(
                    [Receptor("joint", {"left", "right"})], {"joint": torch.zeros(2)}
                )
                converter = manager.create_runtime_snapshot_converter(4096)
                with self.assertRaisesRegex((KeyError, ValueError), message):
                    for tensors, last in batches:
                        converter.add(tensors, is_last=last)
                converter.close()

    def test_rejects_wrong_runtime_shape_and_dtype(self):
        for resident in (torch.zeros(2), torch.zeros(1, dtype=torch.float16)):
            with self.subTest(shape=resident.shape, dtype=resident.dtype):
                manager = manager_for([Receptor("out", {"source"})], {"out": resident})
                converter = manager.create_runtime_snapshot_converter(4096)
                with self.assertRaisesRegex(ValueError, "layout|dtype|shape"):
                    converter.add([("source", torch.ones(1))], is_last=True)
                self.assertTrue(
                    torch.equal(manager._weights.global_weights["out"], resident)
                )

    def test_pending_and_receptor_workspace_are_budgeted_before_load(self):
        loaded = []
        receptor = Receptor(
            "joint", {"left", "right"}, lambda ts: loaded.append(True) or torch.cat(ts)
        )
        manager = manager_for([receptor], {"joint": torch.zeros(2)})
        converter = manager.create_runtime_snapshot_converter(4)
        converter.add([("left", torch.ones(1))])
        with self.assertRaisesRegex(MemoryError, "budget|bytes"):
            converter.add([("right", torch.ones(1))], is_last=True)
        self.assertEqual(loaded, [])
        converter.close()

    def test_plain_embedding_uses_source_plus_local_copy_workspace_bound(self):
        ns = {
            "__name__": "rtp_llm.utils.model_weight",
            "torch": torch,
            "logging": logging,
            "traceback": traceback,
            "functools": functools,
            "weakref": weakref,
            "ABC": ABC,
            "abstractmethod": abstractmethod,
            "WeightStyle": SimpleNamespace(NONE=0),
        }
        load_definitions(
            ROOT / "utils/model_weight.py",
            {"identity", "CkptWeightInfo", "sp_neg1"},
            ns,
        )
        ns["W"] = SimpleNamespace(
            lm_head="lm_head", gpt_style_tp_strategy={"embedding": ns["sp_neg1"]}
        )
        ns["__name__"] = "rtp_llm.model_loader.weight_module"
        load_definitions(
            LOADER / "weight_module.py", {"WeightModule", "AtomicWeight"}, ns
        )
        ns["__name__"] = "rtp_llm.device.device_base"
        load_definitions(ROOT / "device/device_base.py", {"DeviceBase"}, ns)
        receptor = ns["AtomicWeight"]("embedding", [ns["CkptWeightInfo"]("source")])
        manager = manager_for(
            [receptor], {"embedding": torch.zeros(8, 1, dtype=torch.bfloat16)}, rank=2
        )
        config = manager._weights_loader.get_load_config()
        config.compute_dtype = torch.bfloat16
        config.exported_device = ns["DeviceBase"].__new__(ns["DeviceBase"])
        converter = manager.create_runtime_snapshot_converter(112)
        source = torch.arange(32, dtype=torch.bfloat16).reshape(8, 4)
        output = converter.add([("source", source)], is_last=True)[0][1]
        self.assertEqual(output.flatten().tolist(), [2, 6, 10, 14, 18, 22, 26, 30])

    def test_closed_or_finished_converter_rejects_further_input(self):
        manager = manager_for([Receptor("out", {"source"})], {"out": torch.zeros(1)})
        converter = manager.create_runtime_snapshot_converter(4096)
        converter.add([("source", torch.ones(1))], is_last=True)
        with self.assertRaises(RuntimeError):
            converter.add([])
        converter.close()
        converter.close()
        with self.assertRaises(RuntimeError):
            converter.add([])

    def test_independent_converters_do_not_share_pending_state(self):
        manager = manager_for(
            [Receptor("joint", {"a", "b"})], {"joint": torch.zeros(2)}
        )
        left = manager.create_runtime_snapshot_converter(4096)
        right = manager.create_runtime_snapshot_converter(4096)
        left.add([("a", torch.tensor([1.0]))])
        right.add([("a", torch.tensor([3.0]))])
        self.assertEqual(
            left.add([("b", torch.tensor([2.0]))], is_last=True)[0][1].tolist(), [1, 2]
        )
        self.assertEqual(
            right.add([("b", torch.tensor([4.0]))], is_last=True)[0][1].tolist(), [3, 4]
        )

    def test_runtime_install_never_reapplies_receptor_transform(self):
        receptor = Receptor("out", {"source"}, lambda ts: ts[0] + 100)
        manager = manager_for([receptor], {"out": torch.zeros(2)})
        manager.verify_hf_updates = True
        source = torch.tensor([7.0, 8.0])
        manager.update_from_runtime_tensors([('[null,"out"]', source)], is_last=True)
        self.assertEqual(manager._weights.global_weights["out"].tolist(), [7, 8])
        self.assertEqual(manager._runtime_verified_weights, 1)
        self.assertNotEqual(
            manager._weights.global_weights["out"].data_ptr(), source.data_ptr()
        )

    def test_runtime_install_checks_full_batch_before_writing(self):
        for bad_key, tensor, message in [
            ('[null,"b"]', torch.ones(3), "shape|layout"),
            ('[null,"b"]', torch.ones(1, dtype=torch.float16), "dtype|layout"),
            ('[null,"unknown"]', torch.ones(1), "unexpected"),
            ('[null,"a"]', torch.ones(1), "duplicate"),
        ]:
            with self.subTest(message=message):
                manager = manager_for(
                    [Receptor("a", {"a"}), Receptor("b", {"b"})],
                    {"a": torch.zeros(1), "b": torch.zeros(1)},
                )
                with self.assertRaisesRegex((KeyError, ValueError), message):
                    manager.update_from_runtime_tensors(
                        [('[null,"a"]', torch.ones(1)), (bad_key, tensor)], is_last=True
                    )
                self.assertEqual(manager._weights.global_weights["a"].item(), 0)
                manager.abort_runtime_update()

    def test_runtime_round_missing_duplicate_failure_and_abort(self):
        manager = manager_for(
            [Receptor("a", {"a"}), Receptor("b", {"b"})],
            {"a": torch.zeros(1), "b": torch.zeros(1)},
        )
        manager._hf_buffer = {"existing": torch.ones(1)}
        manager.update_from_runtime_tensors([('[null,"a"]', torch.ones(1))])
        with self.assertRaisesRegex(KeyError, "duplicate"):
            manager.update_from_runtime_tensors([('[null,"a"]', torch.ones(1))])
        with self.assertRaisesRegex(RuntimeError, "abort|failed"):
            manager.update_from_runtime_tensors(
                [('[null,"b"]', torch.ones(1))], is_last=True
            )
        manager.abort_runtime_update()
        with self.assertRaisesRegex(KeyError, "missing"):
            manager.update_from_runtime_tensors(
                [('[null,"a"]', torch.ones(1))], is_last=True
            )
        manager.abort_runtime_update()
        manager.update_from_runtime_tensors(
            [
                ('[null,"a"]', torch.full((1,), 2.0)),
                ('[null,"b"]', torch.full((1,), 3.0)),
            ],
            is_last=True,
        )
        self.assertEqual(manager._weights.global_weights["a"].item(), 2)
        self.assertEqual(manager._weights.global_weights["b"].item(), 3)
        self.assertEqual(set(manager._hf_buffer), {"existing"})

    def test_runtime_install_verification_checks_resident_destination(self):
        manager = manager_for([Receptor("out", {"source"})], {"out": torch.zeros(1)})
        manager.verify_hf_updates = True
        manager._weights.update_global_weight = lambda **kw: None
        with self.assertRaisesRegex(RuntimeError, "mismatch"):
            manager.update_from_runtime_tensors(
                [('[null,"out"]', torch.ones(1))], is_last=True
            )

    def test_runtime_key_rejects_ambiguous_or_invalid_names(self):
        self.assertEqual(SNAPSHOT.encode_runtime_key(3, 'name,"x'), '[3,"name,\\"x"]')
        self.assertEqual(
            SNAPSHOT.decode_runtime_key('[null,"global"]'), (None, "global")
        )
        for key in (
            '[true,"x"]',
            '[-1,"x"]',
            '[0,""]',
            '[0,"x",1]',
            "garbage",
            '[0, "x"]',
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                SNAPSHOT.decode_runtime_key(key)


def real_linear_receptors():
    ns = {
        "torch": torch,
        "logging": logging,
        "traceback": traceback,
        "functools": functools,
        "weakref": weakref,
        "ABC": ABC,
        "abstractmethod": abstractmethod,
        "WeightStyle": SimpleNamespace(NONE=0),
    }
    load_definitions(ROOT / "utils/model_weight.py", {"identity", "CkptWeightInfo"}, ns)
    load_definitions(LOADER / "weight_module.py", {"WeightModule", "AtomicWeight"}, ns)
    ns["_linear_attn_split_stratey"] = {}
    load_definitions(
        LOADER / "linear_attn_weight.py",
        {"LinearAttnAtomicWeight", "split_qkvz", "split_ba"},
        ns,
    )
    load_definitions(
        ROOT / "models/qwen3_next/qwen3_next_weight.py",
        {"merge_qkvz_transpose_reorder", "merge_ba_transpose_reorder"},
        ns,
    )
    ns["_linear_attn_split_stratey"].update(qkvz=ns["split_qkvz"], ba=ns["split_ba"])
    config = SimpleNamespace(
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        linear_key_head_dim=1,
        linear_value_head_dim=1,
    )
    make_source = ns["CkptWeightInfo"]
    make_receptor = ns["LinearAttnAtomicWeight"]
    return [
        make_receptor(
            "qkvz",
            [make_source("qkv"), make_source("z")],
            functools.partial(
                ns["merge_qkvz_transpose_reorder"], linear_attention_config=config
            ),
            config,
        ),
        make_receptor(
            "ba",
            [make_source("b"), make_source("a")],
            functools.partial(
                ns["merge_ba_transpose_reorder"], linear_attention_config=config
            ),
            config,
        ),
    ]


class ActualReceptorTp4Test(unittest.TestCase):
    setUp = RuntimeSnapshotTest.setUp

    def test_actual_qwen35_qkvz_and_ba_receptors_match_each_tp_rank(self):
        qkv = torch.arange(24.0).reshape(12, 2)
        z = torch.arange(24.0, 32).reshape(4, 2)
        b = torch.arange(40.0, 48).reshape(4, 2)
        a = torch.arange(50.0, 58).reshape(4, 2)
        for rank in range(4):
            with self.subTest(rank=rank):
                manager = manager_for(
                    [], {}, rank=rank, layer_receptors=[real_linear_receptors()]
                )
                manager._weights.weights[0].update(
                    qkvz=torch.zeros(2, 4), ba=torch.zeros(2, 2)
                )
                converter = manager.create_runtime_snapshot_converter(8192)
                self.assertEqual(converter.add([("qkv", qkv), ("b", b)]), [])
                output = dict(converter.add([("a", a), ("z", z)], is_last=True))
                offset = rank * 2
                self.assertEqual(
                    output['[0,"qkvz"]'].tolist(),
                    [
                        [offset, offset + 8, offset + 16, offset + 24],
                        [offset + 1, offset + 9, offset + 17, offset + 25],
                    ],
                )
                self.assertEqual(
                    output['[0,"ba"]'].tolist(),
                    [[offset + 40, offset + 50], [offset + 41, offset + 51]],
                )
                manager.update_from_runtime_tensors(output.items(), is_last=True)
                self.assertTrue(
                    torch.equal(
                        manager._weights.weights[0]["qkvz"], output['[0,"qkvz"]']
                    )
                )


if __name__ == "__main__":
    unittest.main()

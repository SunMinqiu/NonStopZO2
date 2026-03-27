import logging
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

from zo2.trainer.hf_transformers import log_based_checkpoint as log_based_checkpoint_module
from zo2.trainer.hf_transformers.log_based_checkpoint import LogBasedCheckpointCallback
from zo2.trainer.hf_transformers.log_based_shadow import (
    _commit_shadow_state,
    _load_shadow_replica,
    _replay_retained_suffix,
)


class TinyTiedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(8, 4)
        self.lm_head = nn.Linear(4, 8, bias=False)
        self.lm_head.weight = self.embed.weight


class DummyQueue:
    def __init__(self):
        self.items = []

    def put_nowait(self, item):
        self.items.append(item)


class DummyAsyncAnchor:
    def __init__(self, published_step=None, published_state=None):
        self.published_step = published_step if published_step is not None else -1
        self.published_state = published_state

    def consume_latest_published_snapshot(self, min_step_exclusive=-1):
        if self.published_state is None or self.published_step <= min_step_exclusive:
            return None
        state = self.published_state
        self.published_state = None
        return self.published_step, state


def test_shadow_replica_round_trip_preserves_metadata_and_ties(tmp_path):
    weight = torch.randn(8, 4)
    state_dict = OrderedDict(
        [
            ("embed.weight", weight),
            ("lm_head.weight", weight),
        ]
    )
    tied_groups = [["embed.weight", "lm_head.weight"]]
    replica_path = tmp_path / "shadow_latest.safetensors"

    _commit_shadow_state(state_dict, str(replica_path), base_step=7, committed_step=11, tied_groups=tied_groups)

    loaded, base_step, committed_step = _load_shadow_replica(str(replica_path), tied_groups=tied_groups)
    assert base_step == 7
    assert committed_step == 11
    assert torch.equal(loaded["embed.weight"], weight)
    assert loaded["embed.weight"].data_ptr() == loaded["lm_head.weight"].data_ptr()


def test_init_for_resume_writes_shadow_replica_safetensors(tmp_path):
    callback = LogBasedCheckpointCallback(batch_size=0, enable_shadow=True, instant_recover=True)
    callback.output_dir = str(tmp_path)
    callback.shadow_replica_path = callback._shadow_replica_path()

    model = TinyTiedModel()
    ckpt_dir = tmp_path / "checkpoint-3"
    ckpt_dir.mkdir()
    torch.save(
        {
            "zo_update_history": [
                {"step": 1, "seed": 11, "grad": 0.0, "lr": 1e-5, "wd": 0.0, "zo_eps": 1e-3},
                {"step": 2, "seed": 12, "grad": 0.1, "lr": 1e-5, "wd": 0.0, "zo_eps": 1e-3},
            ],
            "base_checkpoint": "__initial__",
            "batch_size": 0,
            "is_full_checkpoint": False,
        },
        ckpt_dir / "optimizer.pt",
    )

    callback._init_for_resume(model, SimpleNamespace(global_step=3), str(ckpt_dir))

    replica_path = Path(callback.shadow_replica_path)
    assert replica_path.exists()
    loaded, base_step, committed_step = _load_shadow_replica(
        str(replica_path),
        tied_groups=callback._tied_weight_groups,
    )
    assert callback.base_checkpoint_path == "__initial__"
    assert len(callback.update_history) == 2
    assert base_step == 3
    assert committed_step == 3
    assert loaded["embed.weight"].data_ptr() == loaded["lm_head.weight"].data_ptr()


def test_update_base_and_shadow_publishes_anchor_and_queues_rebase(tmp_path):
    callback = LogBasedCheckpointCallback(batch_size=2, enable_shadow=True, instant_recover=True)
    callback.output_dir = str(tmp_path)
    callback.anchor_latest_path = callback._anchor_latest_path()
    callback.update_queue = DummyQueue()
    callback._tied_weight_groups = []

    model = TinyTiedModel()
    callback._update_base_and_shadow(model, step=5)

    anchor_path = Path(callback.anchor_latest_path)
    assert anchor_path.exists()
    loaded, base_step, committed_step = _load_shadow_replica(str(anchor_path))
    assert base_step == 5
    assert committed_step == 5
    assert torch.equal(loaded["embed.weight"], model.state_dict()["embed.weight"])
    assert callback.update_queue.items == [
        {"cmd": "rebase", "step": 5, "path": callback.anchor_latest_path}
    ]


def test_on_async_anchor_persisted_trims_by_global_step_without_requeue(tmp_path):
    callback = LogBasedCheckpointCallback(batch_size=4, enable_shadow=True, instant_recover=False)
    callback.output_dir = str(tmp_path)
    callback.anchor_latest_path = callback._anchor_latest_path()
    callback.update_queue = DummyQueue()
    callback.update_history = [
        {"step": 5, "seed": 105, "grad": 0.1, "lr": 1e-3, "wd": 0.0, "zo_eps": 1e-3},
        {"step": 6, "seed": 106, "grad": 0.2, "lr": 1e-3, "wd": 0.0, "zo_eps": 1e-3},
        {"step": 9, "seed": 109, "grad": 0.3, "lr": 1e-3, "wd": 0.0, "zo_eps": 1e-3},
    ]
    callback.base_checkpoint_step = 4

    state_dict = OrderedDict([("embed.weight", torch.randn(8, 4))])
    _commit_shadow_state(state_dict, callback.anchor_latest_path, base_step=6, committed_step=6)

    callback.on_async_anchor_persisted(6, "/tmp/checkpoint-6")

    assert callback.base_checkpoint_path == "/tmp/checkpoint-6"
    assert callback.base_checkpoint_step == 6
    assert callback._base_pending_seed == 106
    assert callback.update_history == [
        {"step": 9, "seed": 109, "grad": 0.3, "lr": 1e-3, "wd": 0.0, "zo_eps": 1e-3}
    ]
    assert callback.update_queue.items == []


def test_on_step_end_consumes_published_anchor_as_active_base_and_only_rebases_once(tmp_path):
    callback = LogBasedCheckpointCallback(batch_size=4, enable_shadow=True, instant_recover=False)
    callback.output_dir = str(tmp_path)
    callback.anchor_latest_path = callback._anchor_latest_path()
    callback.update_queue = DummyQueue()
    callback.base_checkpoint_path = "/tmp/checkpoint-4"
    callback.base_checkpoint_step = 4
    callback.active_base_step = 4
    callback.update_history = [
        {"step": 5, "seed": 105, "grad": 0.1, "lr": 1e-3, "wd": 0.0, "zo_eps": 1e-3},
        {"step": 6, "seed": 106, "grad": 0.2, "lr": 1e-3, "wd": 0.0, "zo_eps": 1e-3},
    ]

    published_state = OrderedDict([("embed.weight", torch.randn(8, 4))])
    callback._async_anchor = DummyAsyncAnchor(6, published_state)

    callback.on_step_end(None, SimpleNamespace(global_step=6), None)

    assert callback.active_base_step == 6
    assert callback.base_checkpoint_step == 4
    assert callback.base_checkpoint_state is published_state
    assert callback._active_base_pending_seed == 106
    assert callback.update_queue.items == [
        {"cmd": "rebase", "step": 6, "path": callback.anchor_latest_path}
    ]

    callback.on_async_anchor_persisted(6, "/tmp/checkpoint-6")

    assert callback.base_checkpoint_path == "/tmp/checkpoint-6"
    assert callback.base_checkpoint_step == 6
    assert callback.update_history == []
    assert callback.update_queue.items == [
        {"cmd": "rebase", "step": 6, "path": callback.anchor_latest_path}
    ]


def test_reconstruct_on_demand_replays_only_suffix_after_active_base(monkeypatch):
    callback = LogBasedCheckpointCallback(batch_size=4, enable_shadow=False, instant_recover=False)
    callback.base_checkpoint_state = OrderedDict([("w", torch.zeros(2))])
    callback.active_base_step = 6
    callback._active_base_pending_seed = 106
    callback._trainable_param_names = ["w"]
    callback.update_history = [
        {"step": 5, "seed": 105, "grad": 0.1, "lr": 1e-3, "wd": 0.0, "zo_eps": 1e-3},
        {"step": 6, "seed": 106, "grad": 0.2, "lr": 1e-3, "wd": 0.0, "zo_eps": 1e-3},
        {"step": 9, "seed": 109, "grad": 0.3, "lr": 1e-3, "wd": 0.0, "zo_eps": 1e-3},
    ]
    callback.trainer = SimpleNamespace(
        model=SimpleNamespace(
            opt=SimpleNamespace(zo_eps=1e-3, rng_device="native", rstate_queue=[])
        )
    )

    captured = {}

    def fake_replay(state, updates, **kwargs):
        captured["updates"] = list(updates)
        captured["kwargs"] = kwargs
        return state

    monkeypatch.setattr(log_based_checkpoint_module, "_replay_updates_on_state", fake_replay)

    reconstructed = callback._reconstruct_on_demand()

    assert torch.equal(reconstructed["w"], torch.zeros(2))
    assert captured["updates"] == [
        {"step": 9, "seed": 109, "grad": 0.3, "lr": 1e-3, "wd": 0.0, "zo_eps": 1e-3}
    ]
    assert captured["kwargs"]["initial_prev_seed"] == 106
    assert captured["kwargs"]["zo2_mode"] is True


def test_replay_retained_suffix_replays_only_steps_after_rebase():
    state = OrderedDict([("w", torch.zeros(4))])
    retained_updates = {
        2: {"step": 2, "seed": 102, "grad": 0.15, "lr": 1e-2, "wd": 0.0, "zo_eps": 1e-3},
        3: {"step": 3, "seed": 103, "grad": 0.25, "lr": 1e-2, "wd": 0.0, "zo_eps": 1e-3},
    }
    expected = OrderedDict((name, tensor.clone()) for name, tensor in state.items())
    z_dict = log_based_checkpoint_module._generate_z_for_one_step(103, ["w"], expected, "native")
    log_based_checkpoint_module._apply_single_update_with_pregenerated_z(
        expected,
        retained_updates[3],
        ["w"],
        z_dict,
        default_zo_eps=1e-3,
        simulate_perturbation=False,
        adam_state=None,
    )

    last_step = _replay_retained_suffix(
        retained_updates,
        2,
        state,
        ["w"],
        "native",
        False,
        1e-3,
        None,
        log_based_checkpoint_module,
        logging.getLogger("test.shadow"),
    )

    assert last_step == 3
    assert torch.equal(state["w"], expected["w"])

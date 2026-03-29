import logging
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

from zo2.config.mezo_adam import MeZOAdamConfig
from zo2.optimizer.mezo_adam.zo import MeZOAdam
from zo2.trainer.hf_transformers import log_based_checkpoint as log_based_checkpoint_module
from zo2.trainer.hf_transformers.log_based_checkpoint import LogBasedCheckpointCallback
from zo2.trainer.hf_transformers.log_based_shadow import (
    _build_adam_flat_layout,
    _build_shadow_flat_layout,
    _close_shadow_bundle_flat_writer,
    _cleanup_rebase_payload_flat,
    _commit_shadow_bundle_flat,
    _commit_shadow_state,
    _init_shadow_bundle_flat_storage,
    _load_rebase_payload_flat,
    _load_shadow_bundle_flat,
    _load_shadow_replica,
    _open_shadow_bundle_flat_writer,
    _read_shadow_flat_header,
    _replay_retained_suffix,
)


class TinyTiedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(8, 4)
        self.lm_head = nn.Linear(4, 8, bias=False)
        self.lm_head.weight = self.embed.weight


class TinyAdamModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(3, 2)


class DummyAdamOpt:
    def __init__(self, adam_state):
        self._adam_state = {
            "m": OrderedDict((name, tensor.clone()) for name, tensor in adam_state["m"].items()),
            "v": OrderedDict((name, tensor.clone()) for name, tensor in adam_state["v"].items()),
            "t": int(adam_state["t"]),
            "betas": tuple(adam_state["betas"]),
            "adam_eps": float(adam_state["adam_eps"]),
        }
        self.betas = self._adam_state["betas"]
        self.adam_eps = self._adam_state["adam_eps"]

    def get_adam_state(self):
        return {
            "m": OrderedDict((name, tensor.clone()) for name, tensor in self._adam_state["m"].items()),
            "v": OrderedDict((name, tensor.clone()) for name, tensor in self._adam_state["v"].items()),
            "t": int(self._adam_state["t"]),
            "betas": tuple(self._adam_state["betas"]),
            "adam_eps": float(self._adam_state["adam_eps"]),
        }


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


def _make_adam_flat_storage(tmp_path, state_dict, trainable_param_names, has_adam=True):
    header_path = tmp_path / "shadow.flat.header.json"
    stem = str(header_path).removesuffix(".header.json")
    storage = {
        "enabled": True,
        "layout": _build_shadow_flat_layout(state_dict),
        "header_path": str(header_path),
        "buffer_paths": (f"{stem}.bin",),
        "has_adam": bool(has_adam),
    }
    if has_adam:
        storage["adam_layout"] = _build_adam_flat_layout(state_dict, trainable_param_names)
        storage["adam_m_buffer_paths"] = (f"{stem}.adam_m.bin",)
        storage["adam_v_buffer_paths"] = (f"{stem}.adam_v.bin",)
    else:
        storage["adam_layout"] = {"entries": [], "total_bytes": 0}
        storage["adam_m_buffer_paths"] = ()
        storage["adam_v_buffer_paths"] = ()
    return storage


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


def test_shadow_bundle_flat_round_trip_preserves_adam_state_and_header(tmp_path):
    state_dict = OrderedDict(
        [
            ("w", torch.arange(6, dtype=torch.float32).view(2, 3)),
            ("b", torch.tensor([1.0, -2.0], dtype=torch.float32)),
        ]
    )
    adam_state = {
        "m": OrderedDict([("w", torch.full((2, 3), 0.25, dtype=torch.float32))]),
        "v": OrderedDict([("w", torch.full((2, 3), 0.5, dtype=torch.float32))]),
        "t": 7,
        "betas": (0.9, 0.999),
        "adam_eps": 1e-8,
    }
    flat_storage = _make_adam_flat_storage(tmp_path, state_dict, ["w"], has_adam=True)

    _init_shadow_bundle_flat_storage(
        state_dict,
        flat_storage,
        base_step=3,
        committed_step=7,
        adam_state=adam_state,
    )

    loaded_state, loaded_adam, base_step, committed_step = _load_shadow_bundle_flat(flat_storage)
    assert base_step == 3
    assert committed_step == 7
    assert torch.equal(loaded_state["w"], state_dict["w"])
    assert torch.equal(loaded_state["b"], state_dict["b"])
    assert loaded_adam["t"] == 7
    assert loaded_adam["betas"] == (0.9, 0.999)
    assert torch.equal(loaded_adam["m"]["w"], adam_state["m"]["w"])
    assert torch.equal(loaded_adam["v"]["w"], adam_state["v"]["w"])

    writer = _open_shadow_bundle_flat_writer(flat_storage)
    try:
        next_state = OrderedDict((name, tensor.clone().add_(1.0)) for name, tensor in state_dict.items())
        next_adam = {
            "m": OrderedDict([("w", torch.full((2, 3), 1.25, dtype=torch.float32))]),
            "v": OrderedDict([("w", torch.full((2, 3), 1.5, dtype=torch.float32))]),
            "t": 8,
            "betas": (0.9, 0.999),
            "adam_eps": 1e-8,
        }
        _commit_shadow_bundle_flat(
            next_state,
            next_adam,
            writer,
            base_step=7,
            committed_step=8,
        )
    finally:
        _close_shadow_bundle_flat_writer(writer)

    header = _read_shadow_flat_header(flat_storage["header_path"])
    assert header["snapshot_state"] == "ready"
    assert header["base_step"] == 7
    assert header["committed_step"] == 8
    assert header["has_adam"] is True
    assert header["adam_t"] == 8

    loaded_state, loaded_adam, base_step, committed_step = _load_shadow_bundle_flat(flat_storage)
    assert base_step == 7
    assert committed_step == 8
    assert torch.equal(loaded_state["w"], next_state["w"])
    assert torch.equal(loaded_adam["m"]["w"], next_adam["m"]["w"])
    assert torch.equal(loaded_adam["v"]["w"], next_adam["v"]["w"])
    assert loaded_adam["t"] == 8


def test_shadow_bundle_flat_load_fails_when_snapshot_not_ready(tmp_path):
    state_dict = OrderedDict([("w", torch.arange(4, dtype=torch.float32).view(2, 2))])
    flat_storage = _make_adam_flat_storage(tmp_path, state_dict, ["w"], has_adam=False)

    _init_shadow_bundle_flat_storage(
        state_dict,
        flat_storage,
        base_step=1,
        committed_step=1,
        adam_state=None,
    )

    header = _read_shadow_flat_header(flat_storage["header_path"])
    header["snapshot_state"] = "writing"
    with open(flat_storage["header_path"], "w") as f:
        import json

        json.dump(header, f)

    try:
        _load_shadow_bundle_flat(flat_storage)
    except RuntimeError as e:
        assert "incomplete" in str(e)
    else:
        raise AssertionError("expected _load_shadow_bundle_flat to fail for non-ready snapshot")


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


def test_update_base_and_shadow_publishes_flat_anchor_bundle_for_adam(tmp_path):
    callback = LogBasedCheckpointCallback(batch_size=2, enable_shadow=True, instant_recover=False)
    callback.output_dir = str(tmp_path)
    callback.use_shadow_flat_commit = True
    callback.shadow_flat_header_path = callback._shadow_flat_header_storage_path()
    callback.shadow_flat_buffer_paths = callback._shadow_flat_buffer_storage_paths()
    callback.shadow_flat_adam_m_buffer_paths = callback._shadow_flat_adam_m_buffer_storage_paths()
    callback.shadow_flat_adam_v_buffer_paths = callback._shadow_flat_adam_v_buffer_storage_paths()
    callback.rebase_payload_dir = callback._rebase_payload_dir_path()
    callback.update_queue = DummyQueue()
    callback._tied_weight_groups = [["embed.weight", "lm_head.weight"]]

    model = TinyTiedModel()
    callback._trainable_param_names = [name for name, p in model.named_parameters() if p.requires_grad]
    model.opt = DummyAdamOpt(
        {
            "m": OrderedDict([("embed.weight", torch.full_like(model.embed.weight.detach(), 0.5))]),
            "v": OrderedDict([("embed.weight", torch.full_like(model.embed.weight.detach(), 0.75))]),
            "t": 11,
            "betas": (0.9, 0.999),
            "adam_eps": 1e-8,
        }
    )

    callback._update_base_and_shadow(model, step=5)

    payload_path = callback.update_queue.items[0]["path"]
    assert Path(payload_path).exists()
    assert callback.update_queue.items[0]["cmd"] == "rebase"
    assert callback.update_queue.items[0]["step"] == 5

    loaded, loaded_adam, base_step, committed_step = _load_rebase_payload_flat(
        payload_path,
        tied_groups=callback._tied_weight_groups,
    )
    assert base_step == 5
    assert committed_step == 5
    assert loaded["embed.weight"].data_ptr() == loaded["lm_head.weight"].data_ptr()
    assert loaded_adam["t"] == 11
    assert torch.equal(
        loaded_adam["m"]["embed.weight"],
        torch.full_like(model.embed.weight.detach(), 0.5),
    )
    _cleanup_rebase_payload_flat(payload_path)


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


def test_mezo_adam_live_update_matches_replay_update_exactly():
    torch.manual_seed(1234)
    model = TinyAdamModel()
    config = MeZOAdamConfig(lr=1e-5, weight_decay=1e-2, eps=1e-3, rng_device="native")
    opt = MeZOAdam(model, config)
    initial_params = OrderedDict((name, param.detach().clone()) for name, param in model.named_parameters())

    init_adam = {
        "m": OrderedDict(
            (name, torch.full_like(param.detach(), 0.125, dtype=torch.float32))
            for name, param in model.named_parameters()
        ),
        "v": OrderedDict(
            (name, torch.full_like(param.detach(), 0.25, dtype=torch.float32))
            for name, param in model.named_parameters()
        ),
        "t": 2,
        "betas": config.betas,
        "adam_eps": config.adam_eps,
    }
    opt.restore_adam_state(init_adam)
    opt.t = 3
    opt.projected_grad = -0.375

    seed = 314159
    opt._reset_rng(seed)
    opt.zo_update(model)

    replay_state = OrderedDict((name, tensor.clone()) for name, tensor in initial_params.items())
    replay_adam = {
        "m": OrderedDict((name, tensor.clone()) for name, tensor in init_adam["m"].items()),
        "v": OrderedDict((name, tensor.clone()) for name, tensor in init_adam["v"].items()),
        "t": 2,
        "betas": config.betas,
        "adam_eps": config.adam_eps,
    }
    update = {
        "step": 3,
        "seed": seed,
        "grad": opt.projected_grad,
        "lr": config.lr,
        "wd": config.weight_decay,
        "zo_eps": config.eps,
    }
    z_dict = log_based_checkpoint_module._generate_z_for_one_step(seed, list(replay_state.keys()), replay_state, "native")
    log_based_checkpoint_module._apply_single_update_with_pregenerated_z(
        replay_state,
        update,
        list(replay_state.keys()),
        z_dict,
        default_zo_eps=config.eps,
        simulate_perturbation=False,
        adam_state=replay_adam,
    )

    live_state = OrderedDict((name, param.detach().clone()) for name, param in model.named_parameters())
    for name in live_state:
        assert torch.equal(live_state[name], replay_state[name])

    live_adam = opt.get_adam_state()
    assert live_adam["t"] == replay_adam["t"]
    for name in live_adam["m"]:
        assert torch.equal(live_adam["m"][name], replay_adam["m"][name])
        assert torch.equal(live_adam["v"][name], replay_adam["v"][name])

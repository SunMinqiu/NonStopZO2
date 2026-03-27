from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

from zo2.trainer.hf_transformers.log_based_checkpoint import LogBasedCheckpointCallback
from zo2.trainer.hf_transformers.log_based_shadow import _commit_shadow_state, _load_shadow_replica


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


def test_on_async_anchor_persisted_trims_by_global_step_and_requeues_shadow(tmp_path):
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
    assert callback.update_queue.items == [
        {"cmd": "rebase", "step": 6, "path": callback.anchor_latest_path}
    ]

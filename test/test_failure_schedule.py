import importlib.util
from pathlib import Path


MODULE_PATH = Path("/home/users/u0001609/NonStopZO2/zo2/trainer/hf_transformers/log_based_failure_injection.py")


def _load_failure_module():
    spec = importlib.util.spec_from_file_location("log_based_failure_injection", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_gpu_fail_steps_supports_list_and_disabled_values():
    mod = _load_failure_module()

    assert mod.parse_gpu_fail_steps("-1") == []
    assert mod.parse_gpu_fail_steps("15,45,80") == [15, 45, 80]
    assert mod.parse_gpu_fail_steps("80, 15,45") == [15, 45, 80]


def test_gpu_failure_simulator_advances_and_triggers_each_step_once():
    mod = _load_failure_module()

    sim = mod.GPUFailureSimulator()
    sim.set_fail_steps("15,45,80")

    assert sim.get_remaining_fail_steps() == [15, 45, 80]
    assert sim.check_and_fail(14, None) is False
    assert sim.check_and_fail(15, None) is True
    assert sim.get_remaining_fail_steps() == [45, 80]

    sim.advance_past(45)
    assert sim.get_remaining_fail_steps() == [80]
    assert sim.check_and_fail(79, None) is False
    assert sim.check_and_fail(80, None) is True
    assert sim.get_remaining_fail_steps() == []

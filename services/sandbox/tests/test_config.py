"""SANDBOX_* settings: defaults, units, and a wrong value stops the service with the variable's name."""

from pathlib import Path

import pytest

from sandbox.config import Limits, SettingError


def test_defaults_are_generous_and_the_container_switches_are_off():
    limits = Limits.from_env({})
    assert limits.scratch_dir == Path("/scratch") and limits.wipe_dirs == () and limits.sweep is False
    assert limits.max_timeout_s == 600 and limits.queue_wait_s == 300
    assert limits.max_input_bytes == 512 * 1024 * 1024 and limits.max_code_bytes == 256 * 1024
    assert limits.max_output_bytes == 256 * 1024 and limits.max_file_bytes == 256 * 1024 * 1024
    assert (limits.max_files, limits.max_input_files, limits.max_processes) == (50, 64, 128)
    assert limits.mpl_cache is None


def test_values_and_lists_are_read_with_their_units():
    limits = Limits.from_env({"SANDBOX_MAX_INPUT_MB": "2", "SANDBOX_MAX_OUTPUT_KB": "3", "SANDBOX_SWEEP": "1",
                              "SANDBOX_WIPE_DIRS": "/tmp, /dev/shm", "SANDBOX_MAX_TIMEOUT_S": "2.5",
                              "SANDBOX_MPL_CACHE": "/opt/matplotlib"})
    assert limits.max_input_bytes == 2 * 1024 * 1024 and limits.max_output_bytes == 3 * 1024
    assert limits.sweep and limits.wipe_dirs == (Path("/tmp"), Path("/dev/shm")) and limits.max_timeout_s == 2.5
    assert limits.mpl_cache == Path("/opt/matplotlib")


@pytest.mark.parametrize("env, name", [
    ({"SANDBOX_MAX_FILES": "many"}, "SANDBOX_MAX_FILES"),
    ({"SANDBOX_MAX_PROCESSES": "2"}, "SANDBOX_MAX_PROCESSES"),
    ({"SANDBOX_SWEEP": "maybe"}, "SANDBOX_SWEEP"),
    ({"SANDBOX_MAX_TIMEOUT_S": "0"}, "SANDBOX_MAX_TIMEOUT_S"),
    ({"SANDBOX_MAX_INPUT_MB": "0.5"}, "SANDBOX_MAX_INPUT_MB"),
])
def test_a_wrong_value_names_its_variable(env, name):
    with pytest.raises(SettingError, match=name):
        Limits.from_env(env)

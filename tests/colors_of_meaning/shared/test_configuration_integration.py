import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
from assertpy import assert_that

PROBE_SCRIPT = """
import os
from colors_of_meaning.shared.configuration import get_application_setting_provider

provider = get_application_setting_provider()
print(f"experiment_config={provider.get('experiment_config')}")
print(f"host={provider.get('host')}")
"""


def _run_probe_with_environment(env: dict) -> subprocess.CompletedProcess:
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as temp_file:
        temp_file.write(PROBE_SCRIPT)
        temp_file_path = temp_file.name

    try:
        return subprocess.run([sys.executable, temp_file_path], env=env, check=True, capture_output=True, text=True)
    finally:
        Path(temp_file_path).unlink(missing_ok=True)


@pytest.mark.integration
def test_should_use_environment_variable_configuration():
    env = os.environ.copy()
    env["APP_EXPERIMENT_CONFIG"] = "configs/from_environment.yaml"
    env["APP_HOST"] = "127.0.0.1"

    result = _run_probe_with_environment(env)

    assert_that(result.stdout).contains("experiment_config=configs/from_environment.yaml")


@pytest.mark.integration
def test_should_use_properties_file_configuration():
    env = os.environ.copy()
    env.pop("APP_EXPERIMENT_CONFIG", None)
    env.pop("APP_HOST", None)

    result = _run_probe_with_environment(env)

    assert_that(result.stdout).contains("experiment_config=configs/base.yaml")

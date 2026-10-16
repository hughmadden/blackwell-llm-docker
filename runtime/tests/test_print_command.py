"""--print-command: the preset's vllm serve process as a docker run command."""

import shlex
import subprocess
import sys

import pytest

from runtime import launcher
from runtime.launcher import (
    ROOT,
    ConfigError,
    make_argv,
    plain_command,
    plain_values,
    resolve,
)

QAD = "local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD"
QAD_REVISION = "96f27f8c4e443be5bc486f8a978e3b9e0d3f7125"


@pytest.fixture(autouse=True)
def recipe_image(monkeypatch):
    """Resolve like the shipped image: vLLM reads ModelOpt CSF recipes and the
    csf_32x2_v1 block encoding."""
    monkeypatch.setattr(launcher, "installed_semantic_a4_prefill", lambda: False)
    monkeypatch.setattr(launcher, "installed_nvfp4_csf_recipes", lambda: True)


def tp2(env=None, **kwargs):
    return resolve(
        "glm53-flash",
        "rtx-pro-6000-pcie",
        preset="glm53-tp2",
        env={"CACHE_MODE": "vram", **(env or {})},
        csf_formats=frozenset({"nvfp4_csf"}),
        csf_encodings=frozenset({"csf", "csf_32x2_v1"}),
        **kwargs,
    )


def parse(script: str):
    """The docker run argv of a printed script, with its variables at their defaults."""
    body = script.split("\ndocker run ", 1)[1]
    body = body.replace("\\\n", " ").replace(
        "`# inherited from the image or the printing environment`", ""
    )
    defaults = {
        "NAME": "glm53-tp2",
        "GPUS": "0,1",
        "PORT": "8000",
        "HF_CACHE": "model-cache",
        "RUNTIME_CACHE": "glm53-tp2-runtime",
        "IMAGE": "img",
    }
    for key, value in defaults.items():
        body = body.replace(f'"${key}"', value).replace(f"${key}", value)
    return shlex.split(body)


def test_hf_layout_csf_checkpoint_gets_the_execution_time_options():
    plan = tp2()
    values = plain_values(plan)
    assert values["model"] == QAD
    assert values["revision"] == QAD_REVISION
    assert values["quantization"] == "modelopt_mixed"
    assert values["load-format"] == "instanttensor"
    assert values["speculative-config"]["revision"] == QAD_REVISION
    # Everything else is the preset's own value.
    kept = {
        k: v for k, v in plan.values.items() if k not in ("quantization", "load-format")
    }
    assert {
        k: v for k, v in values.items() if k not in ("quantization", "load-format")
    } == kept


def test_script_reproduces_the_launcher_environment_and_argv():
    plan = tp2()
    script = plain_command(plan, plain_values(plan), ["a note"], "glm53-tp2")
    assert script.startswith("#!/usr/bin/env bash\n")
    assert "# a note\n" in script
    argv = parse(script)
    assert argv[:3] == ["-d", "--name", "glm53-tp2"]
    assert argv[argv.index("--gpus") + 1] == '"device=0,1"'
    env = {}
    for index, item in enumerate(argv):
        if item == "-e":
            key, _, value = argv[index + 1].partition("=")
            env[key] = value
    assert env == plan.environment
    serve = argv[argv.index("vllm") :]
    expected = make_argv(plain_values(plan), plan.passthrough)
    assert serve[:3] == ["vllm", "serve", QAD]
    assert serve[3:] == expected[5:]
    assert "--port" in serve and serve[serve.index("--port") + 1] == "8000"
    for name in ("nvfp4_csf", "kv-transfer-config", "lmcache"):
        assert name not in script


def test_secrets_are_passed_by_name_only():
    plan = tp2(env={"HF_TOKEN": "hf_" + "x" * 30})
    script = plain_command(plan, plain_values(plan), [], "glm53-tp2")
    assert "hf_xxx" not in script
    if "HF_TOKEN" in plan.environment:
        assert "\n  -e HF_TOKEN \\\n" in script


def test_passthrough_and_port_override_follow_the_plan():
    plan = tp2(argv=["--port", "8123", "--max-num-seqs", "16"])
    script = plain_command(plan, plain_values(plan), [], "glm53-tp2")
    assert "PORT=${PORT:-8123}" in script
    argv = parse(script)
    assert argv[argv.index("--max-num-seqs") + 1] == "16"


def test_container_layout_and_replicas_have_no_plain_command(monkeypatch):
    plan = tp2()
    plan.values["revision"] = (
        "fd660d51d1fc3caae26a4bf31b7451475bbb9bdc"  # pinned manifest (container layout)
    )
    with pytest.raises(ConfigError, match="generates at start"):
        plain_values(plan)
    plan = tp2()
    plan.values["replicas"] = 2
    with pytest.raises(ConfigError, match="REPLICAS"):
        plain_values(plan)
    plan = tp2()
    plan.draft_subfolder = "dflash"
    with pytest.raises(ConfigError, match="drafter"):
        plain_values(plan)


def test_cli_prints_the_vram_form_of_the_default_preset(tmp_path):
    """Outside an image the CLI resolves the preset as written and prints a script."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "runtime.launcher",
            "--preset",
            "glm53-tp2",
            "--print-command",
        ],
        cwd=ROOT.parent,
        capture_output=True,
        text=True,
        env={
            "PATH": "/usr/bin:/bin",
            "PYTHONPATH": str(ROOT.parent),
            "HOME": str(tmp_path),
        },
        check=False,
    )
    assert result.returncode == 0, result.stderr
    script = result.stdout
    assert "CACHE_MODE=vram form" in script
    assert "vllm serve local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD" in script
    assert "--revision " + QAD_REVISION in script
    assert "kv-transfer-config" not in script
    explicit = subprocess.run(
        [
            sys.executable,
            "-m",
            "runtime.launcher",
            "--preset",
            "glm53-tp2",
            "--print-command",
            "--",
            "--cache-mode",
            "lmcache",
            "--no-cache-l2-enabled",
            "--cache-object-tokens",
            "4096",
        ],
        cwd=ROOT.parent,
        capture_output=True,
        text=True,
        env={
            "PATH": "/usr/bin:/bin",
            "PYTHONPATH": str(ROOT.parent),
            "HOME": str(tmp_path),
        },
        check=False,
    )
    assert explicit.returncode == 0, explicit.stderr
    assert "CACHE_MODE=vram form" in explicit.stdout
    assert "kv-transfer-config" not in explicit.stdout
    assert (
        explicit.stdout.split("\ndocker run", 1)[1]
        == script.split("\ndocker run", 1)[1]
    )


def test_cache_mode_removed_drops_both_spellings():
    from runtime.launcher import cache_mode_removed

    assert cache_mode_removed(["--port", "1", "--cache-mode", "lmcache", "--x"]) == [
        "--port",
        "1",
        "--x",
    ]
    assert cache_mode_removed(["--cache-mode=lmcache", "--y", "2"]) == ["--y", "2"]

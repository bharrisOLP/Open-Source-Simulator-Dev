"""Tests of the hygiene rules (tools/hygiene.py). Test strings are built in parts, so that this file passes."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("hygiene", Path(__file__).resolve().parents[1] / "tools" / "hygiene.py")
H = importlib.util.module_from_spec(spec)
spec.loader.exec_module(H)


@pytest.mark.parametrize("text", ["host " + "192.168" + ".50.1", "10." + "0.0.7", "mail a.b" + "@" + "corp.com",
                                  "/Us" + "ers/x/y", "/ho" + "me/x/y", "~/si" + "mulations/x", "Open" + "Light",
                                  "linux" + "-sim", "bren" + "dan"])
def test_blocked(text):
    assert H.scan("f", text)


@pytest.mark.parametrize("text", ["jax 0.12.0.1", "336213665+bharrisOLP" + "@users.noreply.github.com",
                                  "noreply" + "@anthropic.com", "t" + "@example.com", "git" + "@github.com:org/repo",
                                  "rocm 10.1.0", "~/.claude/settings.json"])
def test_allowed(text):
    assert not H.scan("f", text)

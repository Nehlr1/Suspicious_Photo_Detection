"""YAML config -> attribute-accessible nested dict. All tunables live in config.yaml."""
from __future__ import annotations

import re
from pathlib import Path

import yaml


class Cfg(dict):
    """dict with attribute access, applied recursively."""

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e

    __setattr__ = dict.__setitem__


def _wrap(x):
    return Cfg({k: _wrap(v) for k, v in x.items()}) if isinstance(x, dict) else x


def load_config(path: str | Path) -> Cfg:
    with open(path) as f:
        return _wrap(yaml.safe_load(f))


def set_config_values(path: str | Path, values: dict[str, object]) -> None:
    """Rewrite `key: value` lines in place (first match per key), keeping comments and layout."""
    lines = Path(path).read_text().splitlines(keepends=True)
    for key, val in values.items():
        pat = re.compile(rf"^(\s*{re.escape(key)}:\s*)([^#\n]*?)(\s*#.*)?$")
        for i, line in enumerate(lines):
            m = pat.match(line.rstrip("\n"))
            if m:
                lines[i] = f"{m.group(1)}{val}{m.group(3) or ''}\n"
                break
        else:
            raise KeyError(f"{key} not found in {path}")
    Path(path).write_text("".join(lines))

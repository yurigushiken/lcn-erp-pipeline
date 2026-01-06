from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import os

import yaml


@dataclass
class AnalysisConfig:
    dataset: Dict[str, Any]
    selection: Dict[str, Any]
    components: List[str]
    preprocessing: Dict[str, Any]
    roi: Dict[str, Any]
    plots: Dict[str, Any]
    outputs: Dict[str, Any]
    measurement: Dict[str, Any] = field(default_factory=dict)


def load_analysis_config(path: str) -> AnalysisConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    required = [
        "dataset",
        "selection",
        "components",
        "preprocessing",
        "roi",
        "plots",
        "outputs",
    ]
    for key in required:
        if key not in data:
            raise ValueError(f"Missing required key in config: {key}")

    payload = {k: data[k] for k in required}
    if "measurement" in data and isinstance(data["measurement"], dict):
        payload["measurement"] = data["measurement"]
    return AnalysisConfig(**payload)


def load_components_config(repo_root: str) -> Dict[str, Dict[str, Any]]:
    path = os.path.join(repo_root, "configs", "components.yaml")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing components config: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    comps = data.get("components", {}) if isinstance(data, dict) else {}
    return comps


def load_electrodes_config(repo_root: str) -> Dict[str, List[str]]:
    path = os.path.join(repo_root, "configs", "electrodes.yaml")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing electrodes config: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return {str(k): list(v) for k, v in data.items() if isinstance(v, list)}


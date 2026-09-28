"""Save and load extracted features inside a project.

Layout: ``data/derived/features/<audio stem>/<audio stem>.<kind>.npz`` with a
provenance sidecar (``.npz.json``) next to each file, matching how spectrograms
are stored. Arrays are saved uncompressed because activations compress poorly.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from triton.core.io import write_sidecar
from triton.core.project import project_features_dir


def features_root(project_dir: Path) -> Path:
	return project_features_dir(project_dir)


def feature_dir(project_dir: Path, audio_path: Path) -> Path:
	return features_root(project_dir) / audio_path.stem


def feature_path(project_dir: Path, audio_path: Path, kind: str) -> Path:
	return feature_dir(project_dir, audio_path) / f"{audio_path.stem}.{kind}.npz"


def save_feature(
	project_dir: Path,
	audio_path: Path,
	kind: str,
	arrays: dict[str, np.ndarray],
	*,
	options: dict[str, object],
	extra: dict[str, object] | None = None,
) -> Path:
	"""Write ``arrays`` to the feature file for ``kind`` plus its sidecar JSON.

	The options are also embedded in the .npz as ``meta_json`` so the file is
	self-describing if it is copied out of the project.
	"""
	path = feature_path(project_dir, audio_path, kind)
	path.parent.mkdir(parents=True, exist_ok=True)
	meta = {"kind": kind, "source": audio_path.name, "options": options, **(extra or {})}
	np.savez(path, **arrays, meta_json=np.array(json.dumps(meta, sort_keys=True, default=str)))
	write_sidecar(
		path,
		source={"path": str(audio_path.resolve())},
		actions=[{"step": f"features.{kind}", "options": options}],
		extra={
			"features": {name: list(np.asarray(value).shape) for name, value in arrays.items()},
			**(extra or {}),
		},
	)
	return path


def load_feature(path: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
	"""Load a feature file. Returns (arrays, metadata)."""
	with np.load(path, allow_pickle=False) as data:
		arrays = {name: np.asarray(data[name]) for name in data.files if name != "meta_json"}
		meta = json.loads(str(data["meta_json"].item())) if "meta_json" in data.files else {}
	return arrays, meta


def list_feature_files(project_dir: Path) -> list[Path]:
	root = features_root(project_dir)
	return sorted(root.glob("*/*.npz")) if root.exists() else []

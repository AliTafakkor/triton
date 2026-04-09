"""Project lifecycle and storage helpers shared by GUI and CLI."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import re
import tomllib
from pathlib import Path
from typing import Literal

import numpy as np
import soundfile as sf

from triton.core.conversion import resample, to_mono, to_stereo
from triton.core.io import load_audio, save_audio


ChannelMode = Literal["mono", "stereo"]
BabbleSex = Literal["m", "f"]

PROJECT_CONFIG_NAME = "triton.toml"
RECENT_PROJECTS_LIMIT = 8
APP_CONFIG_DIR = Path.home() / ".config" / "triton"
RECENT_PROJECTS_PATH = APP_CONFIG_DIR / "recent_projects.json"
SUPPORTED_AUDIO_SUFFIXES = {".wav", ".flac", ".ogg", ".mp3", ".m4a"}
BABBLE_TALKER_LABEL_RE = re.compile(r"^bab-(?P<sex>[mf])(?P<index>\d+)$")
DEFAULT_SPECTROGRAM_SETTINGS: dict[str, object] = {
	"type": "stft",
	"n_fft": 1024,
	"hop_length": 256,
	"win_length": 1024,
	"window": "hann",
	"n_mels": 128,
	"fmin": 32.7,
	"fmax": 8000.0,
	"power": 2.0,
}


_BIT_DEPTH_TO_SUBTYPE: dict[int, str] = {
	8: "PCM_U8",
	16: "PCM_16",
	24: "PCM_24",
	32: "PCM_32",
}


@dataclass(slots=True)
class Project:
	name: str
	path: Path
	sample_rate: int
	channel_mode: ChannelMode
	bit_depth: int = 16
	file_format: str = "wav"

	@property
	def raw_dir(self) -> Path:
		return project_raw_dir(self.path)

	@property
	def normalized_dir(self) -> Path:
		return project_normalized_dir(self.path)

	@classmethod
	def load(cls, project_dir: Path) -> Project:
		config_path = project_config_path(project_dir)
		if not config_path.exists():
			raise FileNotFoundError(f"No {PROJECT_CONFIG_NAME} found in {project_dir}.")

		parsed = tomllib.loads(config_path.read_text(encoding="utf-8"))
		audio = parsed.get("audio", {})
		project = parsed.get("project", {})
		return cls(
			name=str(project.get("name", project_dir.name)),
			path=project_dir,
			sample_rate=int(audio.get("sample_rate", 16000)),
			channel_mode=str(audio.get("channels", "mono")),
			bit_depth=int(audio.get("bit_depth", 16)),
			file_format=str(audio.get("format", "wav")),
		)

	@classmethod
	def create(
		cls,
		project_dir: Path,
		sample_rate: int,
		channel_mode: ChannelMode,
		bit_depth: int = 16,
		file_format: str = "wav",
		spectrogram_settings: dict[str, object] | None = None,
	) -> Project:
		project_dir.mkdir(parents=True, exist_ok=True)
		initialize_project_tree(project_dir)
		write_project_config(
			project_dir,
			sample_rate=sample_rate,
			channel_mode=channel_mode,
			bit_depth=bit_depth,
			file_format=file_format,
			spectrogram_settings=spectrogram_settings,
		)
		log_project_event(
			project_dir,
			"project_created",
			{
				"sample_rate": int(sample_rate),
				"channel_mode": str(channel_mode),
				"bit_depth": int(bit_depth),
				"file_format": str(file_format),
				"spectrogram_settings": load_project_spectrogram_settings(project_dir),
			},
		)
		return cls.load(project_dir)

	def register_recent(self) -> None:
		register_recent_project(self.path, self.name)

	def list_files(self) -> list[Path]:
		return list_project_files(self.path)

	def add_file(self, filename: str, content: bytes) -> Path:
		return add_project_file(self.path, filename, content)

	def to_dict(self) -> dict[str, object]:
		return {
			"name": self.name,
			"path": str(self.path),
			"sample_rate": self.sample_rate,
			"channel_mode": self.channel_mode,
			"bit_depth": self.bit_depth,
			"file_format": self.file_format,
		}


@dataclass(slots=True)
class Pipeline:
	name: str
	steps: list[str]
	step_options: dict[str, dict[str, object]] = field(default_factory=dict)


@dataclass(slots=True)
class BabbleTalkerGroup:
	label: str
	sex: BabbleSex
	index: int
	files: list[Path]


def project_config_path(project_dir: Path) -> Path:
	return project_dir / PROJECT_CONFIG_NAME


def project_raw_dir(project_dir: Path) -> Path:
	return project_dir / "data" / "raw"


def project_normalized_dir(project_dir: Path) -> Path:
	return project_dir / "data" / "normalized"


def project_derived_dir(project_dir: Path) -> Path:
	return project_dir / "data" / "derived"


def project_log_path(project_dir: Path) -> Path:
	return project_dir / "metadata" / "project.log.jsonl"


def initialize_project_tree(project_dir: Path) -> None:
	(project_dir / "data" / "raw").mkdir(parents=True, exist_ok=True)
	(project_dir / "data" / "normalized").mkdir(parents=True, exist_ok=True)
	(project_dir / "data" / "derived").mkdir(parents=True, exist_ok=True)
	(project_dir / "metadata").mkdir(parents=True, exist_ok=True)


def write_project_config(
	project_dir: Path,
	sample_rate: int,
	channel_mode: ChannelMode,
	bit_depth: int = 16,
	file_format: str = "wav",
	spectrogram_settings: dict[str, object] | None = None,
) -> None:
	merged_spectrogram = dict(DEFAULT_SPECTROGRAM_SETTINGS)
	if spectrogram_settings:
		for key in DEFAULT_SPECTROGRAM_SETTINGS:
			if key in spectrogram_settings:
				merged_spectrogram[key] = spectrogram_settings[key]

	config_text = _serialize_project_config(
		project_name=project_dir.name,
		project_root=project_dir,
		sample_rate=sample_rate,
		channel_mode=channel_mode,
		bit_depth=bit_depth,
		file_format=file_format,
		spectrogram_settings=merged_spectrogram,
		pipelines=[],
	)
	project_config_path(project_dir).write_text(config_text, encoding="utf-8")


def load_project_config(project_dir: Path) -> Project:
	return Project.load(project_dir)


def load_project_spectrogram_settings(project_dir: Path) -> dict[str, object]:
	config_path = project_config_path(project_dir)
	if not config_path.exists():
		raise FileNotFoundError(f"No {PROJECT_CONFIG_NAME} found in {project_dir}.")

	parsed = tomllib.loads(config_path.read_text(encoding="utf-8"))
	raw = parsed.get("spectrogram", {})
	if not isinstance(raw, dict):
		raw = {}

	merged = dict(DEFAULT_SPECTROGRAM_SETTINGS)
	for key in DEFAULT_SPECTROGRAM_SETTINGS:
		if key in raw:
			merged[key] = raw[key]

	return merged


def update_project_spectrogram_settings(project_dir: Path, spectrogram_settings: dict[str, object]) -> None:
	config_path = project_config_path(project_dir)
	if not config_path.exists():
		raise FileNotFoundError(f"No {PROJECT_CONFIG_NAME} found in {project_dir}.")

	parsed = tomllib.loads(config_path.read_text(encoding="utf-8"))
	project = parsed.get("project", {})
	audio = parsed.get("audio", {})
	pipelines = load_project_pipelines(project_dir)

	project_name = str(project.get("name", project_dir.name))
	sample_rate = int(audio.get("sample_rate", 16000))
	channel_mode = str(audio.get("channels", "mono"))
	bit_depth = int(audio.get("bit_depth", 16))
	file_format = str(audio.get("format", "wav"))

	config_text = _serialize_project_config(
		project_name=project_name,
		project_root=project_dir,
		sample_rate=sample_rate,
		channel_mode=channel_mode,
		bit_depth=bit_depth,
		file_format=file_format,
		spectrogram_settings=spectrogram_settings,
		pipelines=pipelines,
	)
	config_path.write_text(config_text, encoding="utf-8")
	log_project_event(
		project_dir,
		"spectrogram_settings_updated",
		{"spectrogram_settings": load_project_spectrogram_settings(project_dir)},
	)


def log_project_event(project_dir: Path, event: str, details: dict[str, object] | None = None) -> None:
	(project_dir / "metadata").mkdir(parents=True, exist_ok=True)
	entry = {
		"timestamp": datetime.now(timezone.utc).isoformat(),
		"event": str(event),
		"details": details or {},
	}
	with project_log_path(project_dir).open("a", encoding="utf-8") as handle:
		handle.write(json.dumps(entry, default=str))
		handle.write("\n")


def read_project_log(project_dir: Path, limit: int = 200) -> list[dict[str, object]]:
	path = project_log_path(project_dir)
	if not path.exists():
		return []

	entries: list[dict[str, object]] = []
	for line in path.read_text(encoding="utf-8").splitlines():
		line = line.strip()
		if not line:
			continue
		try:
			parsed = json.loads(line)
		except json.JSONDecodeError:
			continue
		if isinstance(parsed, dict):
			entries.append(parsed)

	if limit <= 0:
		return entries
	return entries[-limit:]


def load_project_pipelines(project_dir: Path) -> list[Pipeline]:
	config_path = project_config_path(project_dir)
	if not config_path.exists():
		raise FileNotFoundError(f"No {PROJECT_CONFIG_NAME} found in {project_dir}.")

	parsed = tomllib.loads(config_path.read_text(encoding="utf-8"))
	pipeline_items = parsed.get("pipeline", [])
	if isinstance(pipeline_items, dict):
		pipeline_items = [pipeline_items]

	pipelines: list[Pipeline] = []
	for item in pipeline_items:
		name = str(item.get("name", "")).strip()
		if not name:
			continue
		steps_value = item.get("steps", [])
		if not isinstance(steps_value, list):
			steps_value = []
		steps = [str(step).strip() for step in steps_value if str(step).strip()]

		step_options: dict[str, dict[str, object]] = {}
		raw_step_options_json = item.get("step_options_json")
		if isinstance(raw_step_options_json, str) and raw_step_options_json.strip():
			try:
				decoded = json.loads(raw_step_options_json)
			except json.JSONDecodeError:
				decoded = {}
			if isinstance(decoded, dict):
				for key, value in decoded.items():
					if isinstance(value, dict):
						step_options[str(key)] = {str(k): v for k, v in value.items()}

		# Backward/forward compatibility if options are present as TOML inline table.
		raw_step_options = item.get("step_options")
		if isinstance(raw_step_options, dict):
			for key, value in raw_step_options.items():
				if isinstance(value, dict):
					step_options[str(key)] = {str(k): v for k, v in value.items()}

		pipelines.append(Pipeline(name=name, steps=steps, step_options=step_options))

	return pipelines


def save_project_pipelines(project_dir: Path, pipelines: list[Pipeline]) -> None:
	config_path = project_config_path(project_dir)
	if not config_path.exists():
		raise FileNotFoundError(f"No {PROJECT_CONFIG_NAME} found in {project_dir}.")

	parsed = tomllib.loads(config_path.read_text(encoding="utf-8"))
	project = parsed.get("project", {})
	audio = parsed.get("audio", {})

	project_name = str(project.get("name", project_dir.name))
	sample_rate = int(audio.get("sample_rate", 16000))
	channel_mode = str(audio.get("channels", "mono"))
	bit_depth = int(audio.get("bit_depth", 16))
	file_format = str(audio.get("format", "wav"))
	spectrogram_settings = load_project_spectrogram_settings(project_dir)

	config_text = _serialize_project_config(
		project_name=project_name,
		project_root=project_dir,
		sample_rate=sample_rate,
		channel_mode=channel_mode,
		bit_depth=bit_depth,
		file_format=file_format,
		spectrogram_settings=spectrogram_settings,
		pipelines=pipelines,
	)
	config_path.write_text(config_text, encoding="utf-8")
	log_project_event(
		project_dir,
		"pipelines_saved",
		{"count": len(pipelines), "names": [pipeline.name for pipeline in pipelines]},
	)


def create_project(
	project_dir: Path,
	sample_rate: int,
	channel_mode: ChannelMode,
	bit_depth: int = 16,
	file_format: str = "wav",
	spectrogram_settings: dict[str, object] | None = None,
) -> Project:
	return Project.create(
		project_dir,
		sample_rate=sample_rate,
		channel_mode=channel_mode,
		bit_depth=bit_depth,
		file_format=file_format,
		spectrogram_settings=spectrogram_settings,
	)


def load_recent_projects() -> list[dict[str, str]]:
	if not RECENT_PROJECTS_PATH.exists():
		return []

	try:
		return json.loads(RECENT_PROJECTS_PATH.read_text(encoding="utf-8"))
	except json.JSONDecodeError:
		return []


def save_recent_projects(projects: list[dict[str, str]]) -> None:
	APP_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
	RECENT_PROJECTS_PATH.write_text(json.dumps(projects, indent=2), encoding="utf-8")


def register_recent_project(project_dir: Path, project_name: str) -> None:
	entry = {"name": project_name, "path": str(project_dir.resolve())}
	projects = [item for item in load_recent_projects() if item.get("path") != entry["path"]]
	projects.insert(0, entry)
	save_recent_projects(projects[:RECENT_PROJECTS_LIMIT])


def list_normalized_project_files(project_dir: Path) -> list[Path]:
	"""List normalized audio files in the project's data/normalized directory."""
	norm_dir = project_normalized_dir(project_dir)
	if not norm_dir.exists():
		return []
	return sorted(
		[path for path in norm_dir.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_SUFFIXES],
		key=lambda path: path.name.lower(),
	)


def normalize_project_file(project_dir: Path, raw_path: Path, project: Project) -> Path:
	"""Normalize a raw file to the project spec and save to data/normalized/.

	Converts sample rate, channel mode, and bit depth. Output format is determined
	by the project's ``file_format`` setting. The normalized file shares the stem
	of the raw file and uses the project format as its extension.

	Returns:
		Path to the normalized file.
	"""
	norm_dir = project_normalized_dir(project_dir)
	norm_dir.mkdir(parents=True, exist_ok=True)

	ext = f".{project.file_format.lstrip('.')}"
	norm_path = norm_dir / (raw_path.stem + ext)

	audio, orig_sr = load_audio(raw_path, sr=None, mono=False)

	if orig_sr != project.sample_rate:
		audio = resample(audio, orig_sr=orig_sr, target_sr=project.sample_rate)

	if project.channel_mode == "mono":
		audio = to_mono(audio)
	else:
		audio = to_stereo(audio)
		if audio.ndim == 2 and audio.shape[0] == 2:
			audio = audio.T  # (2, N) → (N, 2) for soundfile

	subtype = _BIT_DEPTH_TO_SUBTYPE.get(project.bit_depth, "PCM_16")
	sf.write(norm_path, audio, project.sample_rate, subtype=subtype)

	log_project_event(
		project_dir,
		"file_normalized",
		{
			"raw": raw_path.name,
			"normalized": norm_path.name,
			"sample_rate": project.sample_rate,
			"channel_mode": project.channel_mode,
			"bit_depth": project.bit_depth,
			"file_format": project.file_format,
		},
	)
	return norm_path


def add_project_file(
	project_dir: Path,
	filename: str,
	content: bytes,
	*,
	filename_prefix: str = "",
) -> Path:
	raw_dir = project_raw_dir(project_dir)
	raw_dir.mkdir(parents=True, exist_ok=True)
	clean_filename = sanitize_filename(filename)
	clean_prefix = sanitize_import_prefix(filename_prefix)
	target_name = f"{clean_prefix}{clean_filename}" if clean_prefix else clean_filename
	target_path = raw_dir / target_name
	target_path.write_bytes(content)
	log_project_event(
		project_dir,
		"file_imported",
		{
			"name": target_path.name,
			"original_name": clean_filename,
			"filename_prefix": clean_prefix,
			"path": str(target_path.resolve()),
			"size_bytes": int(target_path.stat().st_size),
		},
	)
	return target_path


def delete_project_file(file_path: Path) -> None:
	if not (file_path.exists() and file_path.is_file()):
		return
	project_dir = file_path.parents[2] if len(file_path.parents) >= 3 else file_path.parent
	file_name = file_path.name
	file_path.unlink()

	# Also delete the corresponding raw file (any extension with the same stem)
	raw_dir = project_raw_dir(project_dir)
	for raw_candidate in raw_dir.glob(f"{file_path.stem}.*"):
		if raw_candidate.is_file() and raw_candidate.suffix.lower() in SUPPORTED_AUDIO_SUFFIXES:
			raw_candidate.unlink()

	log_project_event(project_dir, "file_deleted", {"name": file_name, "path": str(file_path.resolve())})


def delete_project_files_by_label(project_dir: Path, label: str) -> list[Path]:
	"""Delete all normalized project files that carry a label.

	Returns:
		List of deleted normalized file paths.
	"""
	clean_label = label.strip()
	if not clean_label:
		raise ValueError("Label cannot be empty.")

	files_to_delete = list_project_files(project_dir, label=clean_label)
	if not files_to_delete:
		return []

	all_labels = load_file_labels(project_dir)
	for file_path in files_to_delete:
		delete_project_file(file_path)
		all_labels.pop(file_path.stem, None)
	save_file_labels(project_dir, all_labels)

	log_project_event(
		project_dir,
		"files_deleted_by_label",
		{
			"label": clean_label,
			"count": len(files_to_delete),
			"files": [path.name for path in files_to_delete],
		},
	)
	return files_to_delete


def sanitize_filename(name: str) -> str:
	cleaned = name.strip()
	if not cleaned:
		raise ValueError("File name cannot be empty.")
	if "/" in cleaned or "\\" in cleaned:
		raise ValueError("File name cannot contain path separators.")
	return re.sub(r"\s+", " ", cleaned)


def sanitize_import_prefix(prefix: str) -> str:
	cleaned = re.sub(r"\s+", " ", str(prefix).strip())
	if not cleaned:
		return ""
	if "/" in cleaned or "\\" in cleaned:
		raise ValueError("Filename prefix cannot contain path separators.")
	return cleaned


def rename_project_file(file_path: Path, new_name: str) -> Path:
	clean_name = sanitize_filename(new_name)
	target_path = file_path.with_name(clean_name)
	if target_path.exists() and target_path != file_path:
		raise FileExistsError(f"{clean_name} already exists.")
	if target_path.suffix.lower() not in SUPPORTED_AUDIO_SUFFIXES:
		raise ValueError("Renamed files must keep a supported audio extension.")
	renamed = file_path.rename(target_path)
	project_dir = renamed.parents[2] if len(renamed.parents) >= 3 else renamed.parent

	# Also rename the corresponding raw file (same stem, any audio extension)
	new_stem = target_path.stem
	raw_dir = project_raw_dir(project_dir)
	for raw_candidate in raw_dir.glob(f"{file_path.stem}.*"):
		if raw_candidate.is_file() and raw_candidate.suffix.lower() in SUPPORTED_AUDIO_SUFFIXES:
			raw_candidate.rename(raw_candidate.with_name(new_stem + raw_candidate.suffix))

	# Migrate label key from old stem to new stem
	all_labels = load_file_labels(project_dir)
	if file_path.stem in all_labels:
		all_labels[new_stem] = all_labels.pop(file_path.stem)
		save_file_labels(project_dir, all_labels)

	log_project_event(
		project_dir,
		"file_renamed",
		{
			"old_name": file_path.name,
			"new_name": renamed.name,
			"old_path": str(file_path.resolve()),
			"new_path": str(renamed.resolve()),
		},
	)
	return renamed


def _unique_project_normalized_path(project_dir: Path, filename: str) -> Path:
	"""Return a unique path in the project normalized directory for a generated artifact."""
	norm_dir = project_normalized_dir(project_dir)
	norm_dir.mkdir(parents=True, exist_ok=True)
	target = norm_dir / sanitize_filename(filename)
	if not target.exists():
		return target

	stem = target.stem
	suffix = target.suffix
	counter = 1
	while True:
		candidate = norm_dir / f"{stem}_{counter}{suffix}"
		if not candidate.exists():
			return candidate
		counter += 1


def _unique_project_derived_path(project_dir: Path, filename: str) -> Path:
	"""Return a unique path directly under the project derived directory for a generated artifact."""
	derived_dir = project_derived_dir(project_dir)
	derived_dir.mkdir(parents=True, exist_ok=True)
	target = derived_dir / sanitize_filename(filename)
	if not target.exists():
		return target

	stem = target.stem
	suffix = target.suffix
	counter = 1
	while True:
		candidate = derived_dir / f"{stem}_{counter}{suffix}"
		if not candidate.exists():
			return candidate
		counter += 1


def save_project_generated_audio(
	project_dir: Path,
	filename: str,
	audio: np.ndarray,
	sr: int,
	*,
	label: str | None = None,
	source: dict[str, object] | None = None,
	actions: list[dict[str, object]] | None = None,
	extra: dict[str, object] | None = None,
) -> Path:
	"""Save a generated audio artifact into the project derived directory and label it if requested."""
	target_path = _unique_project_derived_path(project_dir, filename)
	save_audio(target_path, audio, sr, source=source, actions=actions, extra=extra)
	assigned_label = label.strip() if label and label.strip() else None
	if assigned_label is not None:
		set_file_label(project_dir, target_path, assigned_label)

	log_project_event(
		project_dir,
		"project_artifact_added",
		{
			"name": target_path.name,
			"path": str(target_path.resolve()),
			"label": assigned_label,
		},
	)
	return target_path


def _file_labels_path(project_dir: Path) -> Path:
	"""Return path to file labels metadata file."""
	return project_dir / "metadata" / "file_labels.json"


def load_file_labels(project_dir: Path) -> dict[str, list[str]]:
	"""Load file labels from project metadata.

	Returns:
		Dict mapping filenames to lists of label strings.
	"""
	labels_path = _file_labels_path(project_dir)
	if not labels_path.exists():
		return {}

	try:
		raw = json.loads(labels_path.read_text(encoding="utf-8"))
	except (json.JSONDecodeError, IOError):
		return {}

	# Migrate old format (str values) and normalize to list[str]
	# Also migrate old filename keys (with extensions) to stem keys.
	migrated: dict[str, list[str]] = {}
	for k, v in raw.items():
		if isinstance(v, str):
			cleaned = [v.strip()] if v.strip() else []
		elif isinstance(v, list):
			cleaned = [lbl.strip() for lbl in v if isinstance(lbl, str) and lbl.strip()]
		else:
			cleaned = []
		if not cleaned:
			continue
		# Use stem as key; strip audio extension if present
		stem = Path(k).stem if Path(k).suffix.lower() in SUPPORTED_AUDIO_SUFFIXES else k
		existing = migrated.get(stem, [])
		migrated[stem] = existing + [lbl for lbl in cleaned if lbl not in existing]
	return migrated


def save_file_labels(project_dir: Path, labels: dict[str, list[str]]) -> None:
	"""Save file labels to project metadata."""
	labels_path = _file_labels_path(project_dir)
	labels_path.parent.mkdir(parents=True, exist_ok=True)
	labels_path.write_text(json.dumps(labels, indent=2), encoding="utf-8")


def set_file_labels(project_dir: Path, file_path: Path, labels: list[str]) -> None:
	"""Set all labels for a project file (replaces existing labels).

	Args:
		project_dir: Path to the project directory.
		file_path: Path to the audio file (raw or normalized).
		labels: Labels to assign. Empty list removes all labels.
	"""
	norm_dir = project_normalized_dir(project_dir)
	raw_dir = project_raw_dir(project_dir)
	resolved_parent = file_path.resolve().parent
	if resolved_parent not in (norm_dir.resolve(), raw_dir.resolve()):
		raise ValueError(f"File must be in project raw or normalized directory: {file_path}")

	all_labels = load_file_labels(project_dir)
	stem = file_path.stem
	clean = [lbl.strip() for lbl in labels if lbl.strip()]

	if clean:
		all_labels[stem] = clean
	else:
		all_labels.pop(stem, None)

	save_file_labels(project_dir, all_labels)
	log_project_event(
		project_dir,
		"file_labeled",
		{"stem": stem, "labels": clean},
	)


def set_file_label(project_dir: Path, file_path: Path, label: str) -> None:
	"""Set a single label for a project file, replacing any existing labels.

	Args:
		project_dir: Path to the project directory.
		file_path: Path to the audio file.
		label: Label to assign (empty string to remove all labels).
	"""
	set_file_labels(project_dir, file_path, [label] if label.strip() else [])


def set_project_file_labels(project_dir: Path, file_paths: list[Path], label: str) -> None:
	"""Apply the same label to multiple project files."""
	for file_path in file_paths:
		set_file_label(project_dir, file_path, label)


def get_file_labels(project_dir: Path, file_path: Path) -> list[str]:
	"""Get all labels for a project file."""
	labels = load_file_labels(project_dir)
	return labels.get(file_path.stem, [])


def get_file_label(project_dir: Path, file_path: Path) -> str | None:
	"""Get the first label for a project file, if any."""
	file_labels = get_file_labels(project_dir, file_path)
	return file_labels[0] if file_labels else None


def list_project_files(project_dir: Path, label: str | None = None) -> list[Path]:
	"""List audio files in the project, optionally filtered by label.

	Includes files from both ``data/normalized/`` (imported files converted to
	the project spec) and the top level of ``data/derived/`` (generated
	artifacts such as babble mixes added back to the project).  Pipeline step
	outputs nested inside ``data/derived/pipelines/`` are not included.

	Args:
		project_dir: Path to the project directory.
		label: Optional label to filter by.

	Returns:
		Sorted list of audio file paths.
	"""
	collected: list[Path] = []

	norm_dir = project_normalized_dir(project_dir)
	if norm_dir.exists():
		collected.extend(
			path for path in norm_dir.iterdir()
			if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_SUFFIXES
		)

	derived_dir = project_derived_dir(project_dir)
	if derived_dir.exists():
		collected.extend(
			path for path in derived_dir.iterdir()
			if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_SUFFIXES
		)

	files = sorted(collected, key=lambda path: path.name.lower())

	if label is not None:
		labels = load_file_labels(project_dir)
		files = [f for f in files if label in labels.get(f.stem, [])]

	return files


def _parse_babble_talker_label(label: str | None) -> tuple[BabbleSex, int] | None:
	if not label:
		return None

	match = BABBLE_TALKER_LABEL_RE.match(label.strip())
	if not match:
		return None

	return match.group("sex"), int(match.group("index"))


def load_babble_talker_groups(project_dir: Path) -> list[BabbleTalkerGroup]:
	"""Load talker groups from project labels for babble generation.

	Files labeled with the convention ``bab-f1``, ``bab-m2``, etc. are grouped
	by label. Each group represents one talker and may contain multiple files.
	"""
	labels = load_file_labels(project_dir)
	groups: dict[tuple[BabbleSex, int], list[Path]] = {}

	for file_path in list_project_files(project_dir):
		parsed = None
		for lbl in labels.get(file_path.stem, []):
			parsed = _parse_babble_talker_label(lbl)
			if parsed is not None:
				break
		if parsed is None:
			continue
		groups.setdefault(parsed, []).append(file_path)

	return [
		BabbleTalkerGroup(
			label=f"bab-{sex}{index}",
			sex=sex,
			index=index,
			files=sorted(files, key=lambda path: path.name.lower()),
		)
		for (sex, index), files in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1]))
	]


def select_babble_talker_groups(
	project_dir: Path,
	num_talkers: int,
	num_female_talkers: int | None = None,
	num_male_talkers: int | None = None,
) -> list[BabbleTalkerGroup]:
	"""Select babble talker groups from labeled project files.

	If female and male counts are not provided, the requested number of talkers
	is split as evenly as possible across female and male groups.
	"""
	if num_talkers <= 0:
		raise ValueError("Number of talkers must be positive.")

	groups = load_babble_talker_groups(project_dir)
	female_groups = [group for group in groups if group.sex == "f"]
	male_groups = [group for group in groups if group.sex == "m"]
	available_total = len(female_groups) + len(male_groups)
	if available_total < num_talkers:
		raise ValueError(
			f"Only {available_total} babble-labeled talkers were found, but {num_talkers} were requested."
		)

	auto_balance = num_female_talkers is None and num_male_talkers is None
	if num_female_talkers is None and num_male_talkers is None:
		num_female_talkers = min(num_talkers // 2, len(female_groups))
		num_male_talkers = min(num_talkers - num_female_talkers, len(male_groups))
		while num_female_talkers + num_male_talkers < num_talkers:
			female_room = len(female_groups) - num_female_talkers
			male_room = len(male_groups) - num_male_talkers
			if female_room <= 0 and male_room <= 0:
				raise ValueError(
					"Not enough babble-labeled talkers available to balance the requested count."
				)
			if female_room >= male_room and female_room > 0:
				num_female_talkers += 1
			elif male_room > 0:
				num_male_talkers += 1
			elif female_room > 0:
				num_female_talkers += 1
			else:
				num_male_talkers += 1
	elif num_female_talkers is None:
		num_male_talkers = int(num_male_talkers)
		num_female_talkers = num_talkers - num_male_talkers
	elif num_male_talkers is None:
		num_female_talkers = int(num_female_talkers)
		num_male_talkers = num_talkers - num_female_talkers
	else:
		num_female_talkers = int(num_female_talkers)
		num_male_talkers = int(num_male_talkers)
		if num_female_talkers + num_male_talkers != num_talkers:
			raise ValueError("Female and male talker counts must add up to the total talker count.")

	if num_female_talkers < 0 or num_male_talkers < 0:
		raise ValueError("Talker counts must be non-negative.")

	if auto_balance:
		while num_female_talkers > len(female_groups) and num_male_talkers < len(male_groups):
			num_female_talkers -= 1
			num_male_talkers += 1
		while num_male_talkers > len(male_groups) and num_female_talkers < len(female_groups):
			num_male_talkers -= 1
			num_female_talkers += 1
		if num_female_talkers > len(female_groups) or num_male_talkers > len(male_groups):
			raise ValueError(
				"Not enough babble-labeled talkers available to balance the requested count."
			)
	else:
		if num_female_talkers > len(female_groups) or num_male_talkers > len(male_groups):
			raise ValueError(
				"Not enough babble-labeled talkers available for the requested female/male split."
			)

	selected_female = female_groups[:num_female_talkers]
	selected_male = male_groups[:num_male_talkers]
	selected = selected_female + selected_male

	return sorted(selected, key=lambda group: (group.sex, group.index))


def _toml_string(value: str) -> str:
	return json.dumps(value)


def _toml_array_of_strings(values: list[str]) -> str:
	inner = ", ".join(_toml_string(item) for item in values)
	return f"[{inner}]"


def _serialize_project_config(
	*,
	project_name: str,
	project_root: Path,
	sample_rate: int,
	channel_mode: str,
	bit_depth: int = 16,
	file_format: str = "wav",
	spectrogram_settings: dict[str, object],
	pipelines: list[Pipeline],
) -> str:
	settings = dict(DEFAULT_SPECTROGRAM_SETTINGS)
	for key in DEFAULT_SPECTROGRAM_SETTINGS:
		if key in spectrogram_settings:
			settings[key] = spectrogram_settings[key]

	lines = [
		"[project]",
		f"name = {_toml_string(project_name)}",
		f"root = {_toml_string(str(project_root))}",
		"",
		"[audio]",
		f"sample_rate = {sample_rate}",
		f"channels = {_toml_string(channel_mode)}",
		f"bit_depth = {bit_depth}",
		f"format = {_toml_string(file_format)}",
		"",
		"[storage]",
		'raw = "data/raw"',
		'normalized = "data/normalized"',
		'derived = "data/derived"',
		'metadata = "metadata"',
		"",
		"[spectrogram]",
		f"type = {_toml_string(str(settings['type']))}",
		f"n_fft = {int(settings['n_fft'])}",
		f"hop_length = {int(settings['hop_length'])}",
		f"win_length = {int(settings['win_length'])}",
		f"window = {_toml_string(str(settings['window']))}",
		f"n_mels = {int(settings['n_mels'])}",
		f"fmin = {float(settings['fmin'])}",
		f"fmax = {float(settings['fmax'])}",
		f"power = {float(settings['power'])}",
	]

	for pipeline in pipelines:
		if not pipeline.name.strip():
			continue
		steps = [step for step in pipeline.steps if step.strip()]
		step_options = {
			str(step): value
			for step, value in pipeline.step_options.items()
			if isinstance(step, str) and isinstance(value, dict)
		}
		lines.extend(
			[
				"",
				"[[pipeline]]",
				f"name = {_toml_string(pipeline.name.strip())}",
				f"steps = {_toml_array_of_strings(steps)}",
			]
		)
		if step_options:
			lines.append(f"step_options_json = {_toml_string(json.dumps(step_options, sort_keys=True))}")

	lines.append("")
	return "\n".join(lines)

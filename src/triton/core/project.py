"""Project lifecycle and storage helpers shared by GUI and CLI."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
import tomllib
from pathlib import Path
from typing import Literal


ChannelMode = Literal["mono", "stereo"]

PROJECT_CONFIG_NAME = "triton.toml"
RECENT_PROJECTS_LIMIT = 8
APP_CONFIG_DIR = Path.home() / ".config" / "triton"
RECENT_PROJECTS_PATH = APP_CONFIG_DIR / "recent_projects.json"
SUPPORTED_AUDIO_SUFFIXES = {".wav", ".flac", ".ogg", ".mp3", ".m4a"}
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


@dataclass(slots=True)
class Project:
	name: str
	path: Path
	sample_rate: int
	channel_mode: ChannelMode

	@property
	def raw_dir(self) -> Path:
		return project_raw_dir(self.path)

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
		)

	@classmethod
	def create(cls, project_dir: Path, sample_rate: int, channel_mode: ChannelMode) -> Project:
		project_dir.mkdir(parents=True, exist_ok=True)
		initialize_project_tree(project_dir)
		write_project_config(project_dir, sample_rate=sample_rate, channel_mode=channel_mode)
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
		}


@dataclass(slots=True)
class Pipeline:
	name: str
	steps: list[str]
	step_options: dict[str, dict[str, object]] = field(default_factory=dict)


def project_config_path(project_dir: Path) -> Path:
	return project_dir / PROJECT_CONFIG_NAME


def project_raw_dir(project_dir: Path) -> Path:
	return project_dir / "data" / "raw"


def initialize_project_tree(project_dir: Path) -> None:
	(project_dir / "data" / "raw").mkdir(parents=True, exist_ok=True)
	(project_dir / "data" / "normalized").mkdir(parents=True, exist_ok=True)
	(project_dir / "data" / "derived").mkdir(parents=True, exist_ok=True)
	(project_dir / "metadata").mkdir(parents=True, exist_ok=True)


def write_project_config(project_dir: Path, sample_rate: int, channel_mode: ChannelMode) -> None:
	config_text = _serialize_project_config(
		project_name=project_dir.name,
		project_root=project_dir,
		sample_rate=sample_rate,
		channel_mode=channel_mode,
		spectrogram_settings=DEFAULT_SPECTROGRAM_SETTINGS,
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
	spectrogram_settings = load_project_spectrogram_settings(project_dir)

	config_text = _serialize_project_config(
		project_name=project_name,
		project_root=project_dir,
		sample_rate=sample_rate,
		channel_mode=channel_mode,
		spectrogram_settings=spectrogram_settings,
		pipelines=pipelines,
	)
	config_path.write_text(config_text, encoding="utf-8")


def create_project(project_dir: Path, sample_rate: int, channel_mode: ChannelMode) -> Project:
	return Project.create(project_dir, sample_rate=sample_rate, channel_mode=channel_mode)


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


def list_project_files(project_dir: Path) -> list[Path]:
	raw_dir = project_raw_dir(project_dir)
	if not raw_dir.exists():
		return []

	return sorted(
		[path for path in raw_dir.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_SUFFIXES],
		key=lambda path: path.name.lower(),
	)


def add_project_file(project_dir: Path, filename: str, content: bytes) -> Path:
	raw_dir = project_raw_dir(project_dir)
	raw_dir.mkdir(parents=True, exist_ok=True)
	target_path = raw_dir / filename
	target_path.write_bytes(content)
	return target_path


def delete_project_file(file_path: Path) -> None:
	if file_path.exists() and file_path.is_file():
		file_path.unlink()


def sanitize_filename(name: str) -> str:
	cleaned = name.strip()
	if not cleaned:
		raise ValueError("File name cannot be empty.")
	if "/" in cleaned or "\\" in cleaned:
		raise ValueError("File name cannot contain path separators.")
	return re.sub(r"\s+", " ", cleaned)


def rename_project_file(file_path: Path, new_name: str) -> Path:
	clean_name = sanitize_filename(new_name)
	target_path = file_path.with_name(clean_name)
	if target_path.exists() and target_path != file_path:
		raise FileExistsError(f"{clean_name} already exists.")
	if target_path.suffix.lower() not in SUPPORTED_AUDIO_SUFFIXES:
		raise ValueError("Renamed files must keep a supported audio extension.")
	return file_path.rename(target_path)


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

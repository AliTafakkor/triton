"""Project lifecycle and storage helpers shared by GUI and CLI."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import re
import tomllib
from pathlib import Path
from typing import Literal


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
	def create(
		cls,
		project_dir: Path,
		sample_rate: int,
		channel_mode: ChannelMode,
		spectrogram_settings: dict[str, object] | None = None,
	) -> Project:
		project_dir.mkdir(parents=True, exist_ok=True)
		initialize_project_tree(project_dir)
		write_project_config(
			project_dir,
			sample_rate=sample_rate,
			channel_mode=channel_mode,
			spectrogram_settings=spectrogram_settings,
		)
		log_project_event(
			project_dir,
			"project_created",
			{
				"sample_rate": int(sample_rate),
				"channel_mode": str(channel_mode),
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

	config_text = _serialize_project_config(
		project_name=project_name,
		project_root=project_dir,
		sample_rate=sample_rate,
		channel_mode=channel_mode,
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
	log_project_event(
		project_dir,
		"pipelines_saved",
		{"count": len(pipelines), "names": [pipeline.name for pipeline in pipelines]},
	)


def create_project(
	project_dir: Path,
	sample_rate: int,
	channel_mode: ChannelMode,
	spectrogram_settings: dict[str, object] | None = None,
) -> Project:
	return Project.create(
		project_dir,
		sample_rate=sample_rate,
		channel_mode=channel_mode,
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
	log_project_event(
		project_dir,
		"file_imported",
		{"name": target_path.name, "path": str(target_path.resolve()), "size_bytes": int(target_path.stat().st_size)},
	)
	return target_path


def delete_project_file(file_path: Path) -> None:
	if file_path.exists() and file_path.is_file():
		project_dir = file_path.parents[2] if len(file_path.parents) >= 3 else file_path.parent
		file_name = file_path.name
		resolved = str(file_path.resolve())
		file_path.unlink()
		log_project_event(project_dir, "file_deleted", {"name": file_name, "path": resolved})


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
	renamed = file_path.rename(target_path)
	project_dir = renamed.parents[2] if len(renamed.parents) >= 3 else renamed.parent
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


def _file_labels_path(project_dir: Path) -> Path:
	"""Return path to file labels metadata file."""
	return project_dir / "metadata" / "file_labels.json"


def load_file_labels(project_dir: Path) -> dict[str, str]:
	"""Load file labels from project metadata.
	
	Returns:
		Dict mapping relative file paths to label strings.
	"""
	labels_path = _file_labels_path(project_dir)
	if not labels_path.exists():
		return {}
	
	try:
		return json.loads(labels_path.read_text(encoding="utf-8"))
	except (json.JSONDecodeError, IOError):
		return {}


def save_file_labels(project_dir: Path, labels: dict[str, str]) -> None:
	"""Save file labels to project metadata."""
	labels_path = _file_labels_path(project_dir)
	labels_path.parent.mkdir(parents=True, exist_ok=True)
	labels_path.write_text(json.dumps(labels, indent=2), encoding="utf-8")


def set_file_label(project_dir: Path, file_path: Path, label: str) -> None:
	"""Set a label for a project file.
	
	Args:
		project_dir: Path to the project directory.
		file_path: Path to the audio file.
		label: Label to assign (empty string to remove label).
	"""
	raw_dir = project_raw_dir(project_dir)
	# Ensure we're working with a file in the project
	if not file_path.resolve().parent == raw_dir.resolve():
		raise ValueError(f"File must be in project raw directory: {raw_dir}")
	
	labels = load_file_labels(project_dir)
	filename = file_path.name
	
	if label.strip():
		labels[filename] = label.strip()
	else:
		labels.pop(filename, None)
	
	save_file_labels(project_dir, labels)
	log_project_event(
		project_dir,
		"file_labeled",
		{"filename": filename, "label": label.strip() if label.strip() else None},
	)


def set_project_file_labels(project_dir: Path, file_paths: list[Path], label: str) -> None:
	"""Apply the same label to multiple project files."""
	for file_path in file_paths:
		set_file_label(project_dir, file_path, label)


def get_file_label(project_dir: Path, file_path: Path) -> str | None:
	"""Get the label for a project file, if any."""
	labels = load_file_labels(project_dir)
	return labels.get(file_path.name)


def list_project_files(project_dir: Path, label: str | None = None) -> list[Path]:
	"""List audio files in project, optionally filtered by label.
	
	Args:
		project_dir: Path to the project directory.
		label: Optional label to filter by.
	
	Returns:
		Sorted list of audio file paths.
	"""
	raw_dir = project_raw_dir(project_dir)
	if not raw_dir.exists():
		return []

	files = sorted(
		[path for path in raw_dir.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_SUFFIXES],
		key=lambda path: path.name.lower(),
	)
	
	if label is not None:
		labels = load_file_labels(project_dir)
		files = [f for f in files if labels.get(f.name) == label]
	
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
		parsed = _parse_babble_talker_label(labels.get(file_path.name))
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

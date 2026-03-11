"""Project lifecycle and storage helpers shared by GUI and CLI."""

from __future__ import annotations

from dataclasses import dataclass
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
	config_text = (
		"[project]\n"
		f'name = "{project_dir.name}"\n'
		f'root = "{project_dir}"\n\n'
		"[audio]\n"
		f"sample_rate = {sample_rate}\n"
		f'channels = "{channel_mode}"\n\n'
		"[storage]\n"
		'raw = "data/raw"\n'
		'normalized = "data/normalized"\n'
		'derived = "data/derived"\n'
		'metadata = "metadata"\n'
	)
	project_config_path(project_dir).write_text(config_text, encoding="utf-8")


def load_project_config(project_dir: Path) -> Project:
	return Project.load(project_dir)


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

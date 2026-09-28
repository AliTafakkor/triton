"""Example audio: the HARVARD speech-in-noise corpus (University of Salford).

The files are not bundled with Triton. They are licensed CC BY-NC 4.0, while
Triton is MIT, and they total ~100 MB. Instead they are downloaded from the
frozen Figshare archive and every file is checked against a pinned size and
SHA-256, so a truncated or altered download fails loudly instead of being used.

Source: Demonte, P. (2019). HARVARD corpus Speech Shaped Noise and Speech
Modulated Noise for SIN test. University of Salford. Collection.
https://doi.org/10.17866/rd.salford.c.4700054
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import requests

CITATION = (
	"Demonte, P. (2019). HARVARD corpus Speech Shaped Noise and Speech Modulated Noise "
	"for SIN test. University of Salford. https://doi.org/10.17866/rd.salford.c.4700054"
)
LICENSE = "CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)"
DEFAULT_DIR = Path.home() / ".cache" / "triton" / "examples"


@dataclass(frozen=True)
class ExampleFile:
	name: str
	description: str
	url: str
	size: int
	sha256: str
	doi: str


# All files: 48 kHz, mono, 16-bit PCM, 263.3 s. Checksums verified against the
# MD5s Figshare publishes for each file.
EXAMPLES: tuple[ExampleFile, ...] = (
	ExampleFile(
		"Harvard_speech100.wav",
		"100 HARVARD sentences, one talker",
		"https://ndownloader.figshare.com/files/18017723",
		25278486,
		"c177a7bc647470a266ce4389840958d5acf4a94820531e7169b611396b3ff788",
		"10.17866/rd.salford.9988607.v1",
	),
	ExampleFile(
		"HARVARD100_SSN_52nd_order_lpc.wav",
		"Speech-shaped noise (same long-term spectrum as the speech)",
		"https://ndownloader.figshare.com/files/18017762",
		25278486,
		"90d6711347d413be3df78972848dd1d5e1011051ae373ed948243cd487cfb0bc",
		"10.17866/rd.salford.9988655.v1",
	),
	ExampleFile(
		"HARVARD100_SMN_52nd_order_lpc.wav",
		"Speech-modulated noise (speech-shaped noise following the speech envelope)",
		"https://ndownloader.figshare.com/files/18017795",
		25278486,
		"3d31354f7a715fa821f70d750421cbfaaa5e183c5422ff915e96122fe17d0a9b",
		"10.17866/rd.salford.9988673.v1",
	),
	ExampleFile(
		"white_noise_263s_Matlab10.wav",
		"White noise",
		"https://ndownloader.figshare.com/files/18017726",
		25278486,
		"c4eb82bfb8413f33b52d89a8372d1b310251e23e2b780f4cbfe95c9d8724d2b7",
		"10.17866/rd.salford.9988622.v1",
	),
)


class ChecksumError(RuntimeError):
	"""A downloaded file did not match its pinned size or SHA-256."""


def _sha256(path: Path) -> str:
	digest = hashlib.sha256()
	with open(path, "rb") as handle:
		for block in iter(lambda: handle.read(1024 * 1024), b""):
			digest.update(block)
	return digest.hexdigest()


def verify(path: Path, example: ExampleFile) -> bool:
	"""True if ``path`` exists and matches the pinned size and SHA-256."""
	return path.is_file() and path.stat().st_size == example.size and _sha256(path) == example.sha256


def _download_once(example: ExampleFile, target: Path) -> None:
	tmp_path = target.with_name(target.name + ".part")
	try:
		with requests.get(example.url, stream=True, timeout=60) as response:
			response.raise_for_status()
			with open(tmp_path, "wb") as handle:
				for chunk in response.iter_content(chunk_size=1024 * 1024):
					handle.write(chunk)
		if not verify(tmp_path, example):
			got = tmp_path.stat().st_size
			raise ChecksumError(
				f"{example.name}: downloaded {got} bytes, expected {example.size} bytes with "
				f"SHA-256 {example.sha256[:12]}… (truncated or altered download)"
			)
		tmp_path.replace(target)
	finally:
		tmp_path.unlink(missing_ok=True)


def download_examples(
	dest: Path = DEFAULT_DIR,
	*,
	names: list[str] | None = None,
	retries: int = 2,
	on_progress: Callable[[ExampleFile, str], None] | None = None,
) -> list[Path]:
	"""Download (or reuse) verified example files into ``dest``.

	Files already present and matching their checksum are not downloaded again;
	present but mismatching files are replaced. Each file is retried ``retries``
	times on network or checksum failure before giving up.

	Args:
		dest: Directory to store the files.
		names: Subset of example file names; all files if omitted.
		retries: Extra attempts per file after the first failure.
		on_progress: Called with (example, status) where status is
			"cached", "downloading" or "done".

	Returns:
		Paths of the verified files, in manifest order.

	Raises:
		ValueError: If ``names`` contains an unknown file.
		ChecksumError / requests.RequestException: If a file still fails after retries.
	"""
	selected = list(EXAMPLES)
	if names is not None:
		known = {example.name for example in EXAMPLES}
		unknown = sorted(set(names) - known)
		if unknown:
			raise ValueError(f"Unknown example file(s): {', '.join(unknown)}")
		selected = [example for example in EXAMPLES if example.name in set(names)]

	dest = Path(dest).expanduser()
	dest.mkdir(parents=True, exist_ok=True)
	paths: list[Path] = []
	for example in selected:
		target = dest / example.name
		if verify(target, example):
			if on_progress:
				on_progress(example, "cached")
			paths.append(target)
			continue

		if on_progress:
			on_progress(example, "downloading")
		for attempt in range(retries + 1):
			try:
				_download_once(example, target)
				break
			except (ChecksumError, requests.RequestException):
				if attempt == retries:
					raise
		if on_progress:
			on_progress(example, "done")
		paths.append(target)
	return paths

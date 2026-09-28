from __future__ import annotations

import hashlib

import pytest
import requests
from typer.testing import CliRunner

import triton.examples as examples
from triton.cli.main import app

GOOD = b"RIFF" + b"\x01" * 996
FILE = examples.ExampleFile(
	name="test.wav",
	description="test",
	url="https://example.invalid/test.wav",
	size=len(GOOD),
	sha256=hashlib.sha256(GOOD).hexdigest(),
	doi="",
)


class _FakeResponse:
	def __init__(self, body: bytes):
		self.body = body

	def __enter__(self):
		return self

	def __exit__(self, *exc):
		return False

	def raise_for_status(self):
		pass

	def iter_content(self, chunk_size):
		for start in range(0, len(self.body), chunk_size):
			yield self.body[start:start + chunk_size]


@pytest.fixture
def served(monkeypatch):
	"""Queue up response bodies (or exceptions) for successive downloads."""
	queue: list[object] = []
	calls: list[str] = []

	def fake_get(url, stream, timeout):
		calls.append(url)
		item = queue.pop(0)
		if isinstance(item, Exception):
			raise item
		return _FakeResponse(item)

	monkeypatch.setattr(examples.requests, "get", fake_get)
	monkeypatch.setattr(examples, "EXAMPLES", (FILE,))
	return queue, calls


def test_download_verifies_and_writes_file(tmp_path, served) -> None:
	queue, calls = served
	queue.append(GOOD)

	paths = examples.download_examples(tmp_path)

	assert paths == [tmp_path / "test.wav"]
	assert paths[0].read_bytes() == GOOD
	assert len(calls) == 1
	assert list(tmp_path.glob("*.part")) == []


def test_verified_file_is_not_downloaded_again(tmp_path, served) -> None:
	queue, calls = served
	(tmp_path / "test.wav").write_bytes(GOOD)
	statuses = []

	examples.download_examples(tmp_path, on_progress=lambda example, status: statuses.append(status))

	assert calls == []
	assert statuses == ["cached"]


def test_truncated_download_is_retried_then_succeeds(tmp_path, served) -> None:
	queue, calls = served
	queue.extend([GOOD[:500], GOOD])

	paths = examples.download_examples(tmp_path, retries=1)

	assert len(calls) == 2
	assert paths[0].read_bytes() == GOOD


def test_truncated_download_fails_loudly_and_leaves_nothing(tmp_path, served) -> None:
	queue, calls = served
	queue.extend([GOOD[:500], GOOD[:500]])

	with pytest.raises(examples.ChecksumError, match="truncated or altered"):
		examples.download_examples(tmp_path, retries=1)

	assert not (tmp_path / "test.wav").exists()
	assert list(tmp_path.glob("*.part")) == []


def test_corrupted_existing_file_is_replaced(tmp_path, served) -> None:
	queue, calls = served
	(tmp_path / "test.wav").write_bytes(b"X" * len(GOOD))  # right size, wrong bytes
	queue.append(GOOD)

	examples.download_examples(tmp_path)

	assert (tmp_path / "test.wav").read_bytes() == GOOD


def test_network_error_is_retried(tmp_path, served) -> None:
	queue, calls = served
	queue.extend([requests.ConnectionError("boom"), GOOD])

	examples.download_examples(tmp_path, retries=1)

	assert len(calls) == 2


def test_unknown_name_is_rejected(tmp_path, served) -> None:
	with pytest.raises(ValueError, match="nope.wav"):
		examples.download_examples(tmp_path, names=["nope.wav"])


def test_cli_exits_nonzero_on_bad_download(tmp_path, served) -> None:
	queue, calls = served
	queue.extend([GOOD[:10]] * 3)

	result = CliRunner().invoke(app, ["examples", "download", "--dest", str(tmp_path)])

	assert result.exit_code == 1
	assert "Download failed" in result.output


def test_manifest_is_well_formed() -> None:
	names = [example.name for example in examples.EXAMPLES]
	assert len(names) == len(set(names)) == 4
	for example in examples.EXAMPLES:
		assert len(example.sha256) == 64
		assert example.url.startswith("https://ndownloader.figshare.com/files/")

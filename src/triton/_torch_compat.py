"""Workaround for this package sharing its import name with NVIDIA's CUDA ``triton``."""

from __future__ import annotations

import importlib.util


def _has_cuda_triton() -> bool:
	"""True only if the importable ``triton`` is NVIDIA's (it has ``triton.language``)."""
	try:
		return importlib.util.find_spec("triton.language") is not None
	except (ImportError, ValueError):
		return False


def patch_torch_compiler() -> None:
	"""Stop torch from mistaking this package for NVIDIA's ``triton``.

	torch decides whether the CUDA compiler is installed with a bare
	``import triton`` (``torch.utils._triton.has_triton_package``). That import
	finds this package, so torch then reaches for ``triton.language`` and crashes
	the first time anything imports ``torch._dynamo``: model loading in
	transformers (weight_norm, ``@torch.compiler.disable``) does this. Replacing
	the check with one that looks for ``triton.language`` makes torch correctly
	conclude the CUDA compiler is absent. Call before importing ``transformers``;
	safe to call more than once.
	"""
	import torch.utils._triton as torch_triton

	torch_triton.has_triton_package = _has_cuda_triton

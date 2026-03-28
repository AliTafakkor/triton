"""Audio degradation utilities."""

from triton.degrade.noise_generator import compute_ltass, generate_babble, generate_ssn
from triton.degrade.noise_mixer import add_noise
from triton.degrade.vocoder import noise_vocode
from triton.degrade.time_compression import compress_time

__all__ = [
	"compute_ltass",
	"generate_babble",
	"generate_ssn",
	"add_noise",
	"noise_vocode",
	"compress_time",
]

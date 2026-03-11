"""Audio degradation utilities."""

from triton.degrade.noise_generator import compute_ltass, generate_babble, generate_ssn
from triton.degrade.noise_mixer import add_noise
from triton.degrade.vocoder import noise_vocode

__all__ = [
	"compute_ltass",
	"generate_babble",
	"generate_ssn",
	"add_noise",
	"noise_vocode",
]

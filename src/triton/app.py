"""Triton Streamlit app."""

from __future__ import annotations

from io import BytesIO

import librosa
import numpy as np
import soundfile as sf
import streamlit as st

from triton.core.mixer import mix_at_snr


def _load_uploaded_audio(uploaded_file, target_sr: int | None = None) -> tuple[np.ndarray, int]:
	audio_bytes = uploaded_file.getvalue()
	audio, sr = librosa.load(BytesIO(audio_bytes), sr=target_sr, mono=True)
	return audio, sr


def _audio_bytes(audio: np.ndarray, sr: int) -> bytes:
	buffer = BytesIO()
	sf.write(buffer, audio, sr, format="WAV")
	buffer.seek(0)
	return buffer.read()


st.set_page_config(page_title="Triton", page_icon="🐚", layout="wide")

st.markdown(
	"""
	<style>
	.main { background: linear-gradient(180deg, #0b1d2a 0%, #102a43 40%, #123f57 100%); }
	h1, h2, h3, h4, h5, h6, p, label, div { color: #e6f1ff !important; }
	.stButton>button { background-color: #0ea5b7; color: white; border: none; }
	.stSlider>div>div { color: #e6f1ff; }
	.stFileUploader { background-color: #0f2d3f; border-radius: 8px; padding: 8px; }
	.stAudio { background-color: #0f2d3f; border-radius: 8px; padding: 8px; }
	</style>
	""",
	unsafe_allow_html=True,
)

st.title("Triton 🐚")
st.subheader("Nautical audio mixing dashboard for the CONCH Lab")
st.write("Set sail with speech-in-noise mixing at precise SNRs.")

col1, col2 = st.columns(2)
with col1:
	speech_file = st.file_uploader("Upload speech audio", type=["wav", "flac", "ogg", "mp3", "m4a"])
with col2:
	noise_file = st.file_uploader("Upload noise audio", type=["wav", "flac", "ogg", "mp3", "m4a"])

snr_db = st.slider("Target SNR (dB)", min_value=-30.0, max_value=30.0, value=-5.0, step=0.5)

mix_button = st.button("Mix and Play")

if mix_button:
	if not speech_file or not noise_file:
		st.error("Please upload both speech and noise files.")
	else:
		with st.spinner("Mixing at sea..."):
			speech_audio, speech_sr = _load_uploaded_audio(speech_file)
			noise_audio, _ = _load_uploaded_audio(noise_file, target_sr=speech_sr)

			mixed = mix_at_snr(speech_audio, noise_audio, snr_db)
			mixed_bytes = _audio_bytes(mixed, speech_sr)

		st.success("Mix complete.")
		st.audio(mixed_bytes, format="audio/wav")

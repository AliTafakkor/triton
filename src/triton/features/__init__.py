"""Time-aligned feature extraction for relating audio to brain recordings.

Each extractor returns a ``times`` vector (seconds) alongside its values so that
features computed at different rates can be aligned with each other and with
EEG/MEG data downstream.
"""

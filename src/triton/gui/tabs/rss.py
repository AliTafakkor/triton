"""RSS ingest tab for the Triton GUI."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path

import streamlit as st

from triton.core.project import Project, log_project_event, normalize_project_file, project_raw_dir
from triton.ingest.rss import RssSource
from triton.gui.shared import _generate_file_spectrogram


def _parse_episode_published_date(published: str | None) -> date | None:
	if not published:
		return None

	text = published.strip()
	if not text:
		return None

	try:
		if text.endswith("Z"):
			parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
		else:
			parsed = datetime.fromisoformat(text)
		return parsed.date()
	except ValueError:
		pass

	try:
		return parsedate_to_datetime(text).date()
	except (TypeError, ValueError, IndexError):
		return None


def _render_rss_ingest_tab(project: Project) -> None:
	st.markdown("### Podcast RSS Ingest")
	st.write(
		"Fetch podcast episodes from an RSS feed and add them into this project's raw storage. "
		"Each downloaded file also gets a spectrogram artifact."
	)

	raw_dir = project_raw_dir(project.path)
	st.caption(f"Downloads will be saved to {raw_dir} then normalized to the project spec")

	with st.form("rss_ingest_form"):
		default_end = datetime.now().date()
		default_start = default_end - timedelta(days=30)
		if "rss_ingest_start_date" not in st.session_state:
			st.session_state["rss_ingest_start_date"] = default_start
		if "rss_ingest_end_date" not in st.session_state:
			st.session_state["rss_ingest_end_date"] = default_end

		feed_url = st.text_input(
			"Podcast RSS URL",
			placeholder="https://example.com/feed.xml",
			key="rss_ingest_feed_url",
		)
		controls_col1, controls_col2 = st.columns(2)
		with controls_col1:
			limit = int(st.number_input("Max episodes", min_value=1, max_value=200, value=10, step=1, key="rss_ingest_limit"))
		with controls_col2:
			overwrite = st.checkbox("Overwrite existing files", value=False, key="rss_ingest_overwrite")

		use_date_range = st.checkbox("Filter by publish date", value=False, key="rss_ingest_use_date_range")
		date_preset = st.selectbox(
			"Date preset",
			options=["Custom", "Last 7 days", "Last 30 days", "Last 90 days"],
			key="rss_ingest_date_preset",
			disabled=not use_date_range,
		)

		if use_date_range and date_preset != "Custom":
			preset_days = {"Last 7 days": 7, "Last 30 days": 30, "Last 90 days": 90}.get(date_preset, 30)
			st.session_state["rss_ingest_end_date"] = default_end
			st.session_state["rss_ingest_start_date"] = default_end - timedelta(days=preset_days - 1)

		date_col1, date_col2 = st.columns(2)
		with date_col1:
			start_date = st.date_input("Start date", key="rss_ingest_start_date", disabled=not use_date_range or date_preset != "Custom")
		with date_col2:
			end_date = st.date_input("End date", key="rss_ingest_end_date", disabled=not use_date_range or date_preset != "Custom")

		if use_date_range and date_preset != "Custom":
			st.caption(f"Using preset window: {date_preset.lower()}.")

		fetch_only = st.checkbox("Preview only (do not download)", value=True, key="rss_ingest_preview_only")
		submitted = st.form_submit_button("Fetch RSS", type="primary")

	if not submitted:
		return

	clean_feed_url = feed_url.strip()
	if not clean_feed_url:
		st.error("RSS URL is required.")
		return

	with st.spinner("Reading feed..."):
		try:
			source = RssSource(clean_feed_url)
			entries = source.list_entries()
		except Exception as exc:
			st.error(f"Could not read RSS feed: {exc}")
			return

	if not entries:
		st.warning("No audio enclosure entries were found in this feed.")
		return

	if use_date_range:
		if start_date > end_date:
			st.error("Start date must be earlier than or equal to end date.")
			return

		filtered_entries = []
		missing_dates = 0
		for episode in entries:
			published_date = _parse_episode_published_date(episode.published)
			if published_date is None:
				missing_dates += 1
				continue
			if start_date <= published_date <= end_date:
				filtered_entries.append(episode)

		entries = filtered_entries
		if missing_dates:
			st.caption(f"Skipped {missing_dates} episode(s) without a parseable publish date.")

		if not entries:
			st.warning("No episodes matched the selected date range.")
			return

	selected_entries = entries[: max(0, limit)]
	st.success(f"Found {len(entries)} audio episode(s); showing first {len(selected_entries)}.")

	preview_rows = []
	for episode in selected_entries:
		preview_rows.append({"Title": episode.title, "Published": episode.published or "", "Filename": episode.filename, "URL": episode.url})
	st.dataframe(preview_rows, width="stretch")

	if fetch_only:
		return

	with st.status("Downloading episodes...", expanded=True) as download_status:
		try:
			downloaded_paths = source.download(selected_entries, raw_dir, overwrite=overwrite)
		except Exception as exc:
			st.error(f"RSS download failed: {exc}")
			return

	generated_specs = 0
	spec_errors: list[str] = []

	if downloaded_paths:
		with st.status("Normalizing and generating spectrograms...", expanded=True) as spec_status:
			spec_progress_bar = st.progress(0.0)
			for idx, path_str in enumerate(downloaded_paths):
				raw_file = Path(path_str)
				spec_status.write(f"Processing: {raw_file.name}")
				try:
					norm_file = normalize_project_file(project.path, raw_file, project)
					_generate_file_spectrogram(norm_file, project)
				except Exception as exc:
					spec_errors.append(f"{raw_file.name}: {exc}")
				else:
					generated_specs += 1
				spec_progress_val = (idx + 1) / len(downloaded_paths)
				spec_status.update(label=f"Processing... ({idx + 1}/{len(downloaded_paths)})", state="running")
				spec_progress_bar.progress(spec_progress_val)
			spec_status.update(label=f"Normalized and generated {generated_specs} spectrogram(s)", state="complete")

	log_project_event(project.path, "rss_ingest_completed", {"feed_url": clean_feed_url, "requested_entries": len(selected_entries), "downloaded_files": len(downloaded_paths), "spectrograms_generated": generated_specs, "spectrogram_failures": len(spec_errors)})

	st.success(f"Downloaded and normalized {len(downloaded_paths)} file(s).")
	if spec_errors:
		st.warning(f"Generated {generated_specs} spectrogram(s), {len(spec_errors)} failed.")
		for error in spec_errors:
			st.caption(error)
	st.rerun()

"""File library tab for the Triton GUI."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from triton.core.project import Project, load_file_labels, project_normalized_dir, set_file_labels
from triton.core.spectrogram import load_spectrogram
from triton.gui.shared import (
	_delete_project_files_by_label,
	_delete_project_file,
	_format_file_size,
	_generate_file_spectrogram,
	_pipeline_key,
	_save_uploaded_project_files,
	_spectrogram_path,
)


def _on_label_change(label_key: str, project_path: Path, file_path: Path) -> None:
	"""on_change callback for label text inputs — saves immediately when user commits."""
	new_label_str = st.session_state.get(label_key, "")
	new_labels = [lbl.strip() for lbl in new_label_str.split(",") if lbl.strip()]
	set_file_labels(project_path, file_path, new_labels)


def _render_file_library(project: Project, project_files: list[Path]) -> None:
	st.markdown("### Import Files")
	st.caption(f"Normalized files are stored in {project_normalized_dir(project.path)}")

	add_col, count_col = st.columns([2, 3])
	with add_col:
		with st.form("project_file_upload_form"):
			uploaded_files = st.file_uploader(
				"Import audio files",
				type=["wav", "flac", "ogg", "mp3", "m4a"],
				accept_multiple_files=True,
				key="project_file_upload",
			)
			filename_prefix = st.text_input(
				"Filename prefix (optional)",
				value="",
				placeholder="e.g., feed1_",
				key="project_file_upload_prefix",
				help="Prepended to every imported filename to avoid name collisions.",
			)
			batch_label = st.text_input(
				"Apply one label to all imported files",
				value="",
				placeholder="e.g., bab-f1",
				key="project_file_upload_label",
			)
			upload_submitted = st.form_submit_button("Import selected files", type="primary")

		if upload_submitted:
			if not uploaded_files:
				st.error("Choose at least one file to import.")
			else:
				saved_paths = _save_uploaded_project_files(
					project,
					list(uploaded_files),
					batch_label=batch_label,
					filename_prefix=filename_prefix,
				)
				st.success(f"Imported {len(saved_paths)} file(s) and precomputed spectrograms.")
				st.session_state.pop("project_file_upload_label", None)
				st.session_state.pop("project_file_upload_prefix", None)
				st.session_state.pop("selected_spectrogram_file", None)
				st.session_state.pop("file_list_page", None)
				st.session_state.pop("project_file_upload_form", None)
				for key in list(st.session_state.keys()):
					if key.startswith("import_checked_"):
						st.session_state.pop(key, None)
				st.rerun()

	with count_col:
		st.markdown(
			f"""
			<div style="height: 100%; min-height: 136px; padding: 18px 20px; border-radius: 22px; background: var(--card-top-bg); border: 1px solid var(--panel-border);">
			  <div style="font-size: 12px; text-transform: uppercase; letter-spacing: 0.14em; color: var(--hero-kicker); margin-bottom: 8px;">Import status</div>
			  <div style="font-size: 36px; font-weight: 800; margin-bottom: 6px;">{len(project_files)}</div>
			  <div style="color: var(--card-subtle-text);">file(s) available for playback, spectrogram, and pipelines.</div>
			</div>
			""",
			unsafe_allow_html=True,
		)

	if not project_files:
		st.info("No files have been imported to this project yet.")
		return

	selected_spectrogram = st.session_state.get("selected_spectrogram_file")

	st.markdown("### Imported Files")
	list_col, panel_col = st.columns([1.9, 1.1], gap="large")

	with list_col:
		filter_row_col1, filter_row_col2, filter_row_col3 = st.columns([2, 2, 2])

		with filter_row_col1:
			search_text = st.text_input("Search files", value="", placeholder="Type a file name...")

		with filter_row_col2:
			all_labels = load_file_labels(project.path)
			available_labels = sorted(set(lbl for lbls in all_labels.values() for lbl in lbls))
			label_filter = st.selectbox(
				"Filter by label",
				options=["(All)", "(None)"] + available_labels,
				format_func=lambda x: x if x in ("(All)", "(None)") else f"Label: {x}"
			)

		with filter_row_col3:
			sort_mode = st.selectbox("Sort", options=["name", "size_desc", "size_asc"], format_func=lambda value: {
				"name": "Name (A-Z)",
				"size_desc": "Size (largest first)",
				"size_asc": "Size (smallest first)",
			}[value])

		if available_labels:
			st.markdown("##### Bulk Actions")
			delete_col1, delete_col2, delete_col3 = st.columns([2.4, 0.9, 1.2])
			with delete_col1:
				label_to_delete = st.selectbox(
					"Delete all files with label",
					options=["(Select label)"] + available_labels,
					key="bulk_delete_label_select",
				)
			with delete_col2:
				confirm_delete = st.checkbox(
					"Confirm",
					key="bulk_delete_label_confirm",
					help="Required to prevent accidental bulk deletion.",
				)
			with delete_col3:
				delete_disabled = label_to_delete == "(Select label)" or not confirm_delete
				if st.button("Delete Label Files", type="secondary", use_container_width=True, disabled=delete_disabled):
					deleted_files = _delete_project_files_by_label(project.path, label_to_delete)
					deleted_set = {str(path) for path in deleted_files}
					for deleted in deleted_files:
						st.session_state.pop(f"import_checked_{_pipeline_key(str(deleted))}", None)
					if st.session_state.get("selected_spectrogram_file") in deleted_set:
						st.session_state.pop("selected_spectrogram_file", None)
					st.session_state.pop("bulk_delete_label_confirm", None)
					st.success(f"Deleted {len(deleted_files)} file(s) with label '{label_to_delete}'.")
					st.rerun()

		visible_files = [path for path in project_files if search_text.strip().lower() in path.name.lower()]

		if label_filter == "(None)":
			visible_files = [f for f in visible_files if not all_labels.get(f.stem)]
		elif label_filter != "(All)":
			visible_files = [f for f in visible_files if label_filter in all_labels.get(f.stem, [])]

		if sort_mode == "size_desc":
			visible_files = sorted(visible_files, key=lambda path: path.stat().st_size, reverse=True)
		elif sort_mode == "size_asc":
			visible_files = sorted(visible_files, key=lambda path: path.stat().st_size)

		if not visible_files:
			st.info("No files match your search.")
		else:
			items_per_page = 15
			total_pages = (len(visible_files) + items_per_page - 1) // items_per_page
			current_page = st.session_state.get("file_list_page", 0)
			current_page = min(current_page, total_pages - 1)

			start_idx = current_page * items_per_page
			end_idx = start_idx + items_per_page
			page_files = visible_files[start_idx:end_idx]

			header_cols = st.columns([0.4, 2.0, 1.5, 1.0, 1.2, 0.7, 0.7, 0.7])
			with header_cols[0]:
				st.caption("✓")
			with header_cols[1]:
				st.caption("**File Name**")
			with header_cols[2]:
				st.caption("**Size**")
			with header_cols[3]:
				st.caption("**Type**")
			with header_cols[4]:
				st.caption("**Label**")
			with header_cols[5]:
				st.caption("**View**")
			with header_cols[6]:
				st.caption("**Rename**")
			with header_cols[7]:
				st.caption("**Delete**")

			for index, file_path in enumerate(page_files):
				global_index = start_idx + index
				spec_path = _spectrogram_path(file_path)
				check_key = f"import_checked_{_pipeline_key(str(file_path))}"
				file_label_str = ", ".join(all_labels.get(file_path.stem, []))

				row_cols = st.columns([0.4, 2.0, 1.5, 1.0, 1.2, 0.7, 0.7, 0.7])

				with row_cols[0]:
					st.checkbox("Select file", key=check_key, label_visibility="collapsed")

				with row_cols[1]:
					st.caption(file_path.name)

				with row_cols[2]:
					st.caption(_format_file_size(file_path.stat().st_size))

				with row_cols[3]:
					st.caption(file_path.suffix.lower())

				with row_cols[4]:
					label_key = f"label_input_{_pipeline_key(str(file_path))}"
					# Sync session state to disk value so external changes (rename, batch) are reflected
					st.session_state[label_key] = file_label_str
					st.text_input(
						"Set label",
						key=label_key,
						label_visibility="collapsed",
						placeholder="e.g., talker1, bab-f1",
						on_change=_on_label_change,
						args=(label_key, project.path, file_path),
					)

				with row_cols[5]:
					if st.button("📊", key=f"spec_{global_index}", help="View spectrogram", use_container_width=True):
						if not spec_path.exists():
							try:
								_generate_file_spectrogram(file_path, project)
							except Exception as exc:
								st.error(f"Could not generate spectrogram for {file_path.name}: {exc}")
							else:
								st.session_state["selected_spectrogram_file"] = str(file_path)
								st.rerun()
						else:
							st.session_state["selected_spectrogram_file"] = str(file_path)
							st.rerun()

				with row_cols[6]:
					if st.button("✏️", key=f"rename_{global_index}", help="Rename file", use_container_width=True):
						st.session_state["rename_mode"] = global_index
						st.rerun()

				with row_cols[7]:
					if st.button("🗑️", key=f"delete_{global_index}", help="Delete file", use_container_width=True):
						_delete_project_file(file_path)
						if st.session_state.get("selected_spectrogram_file") == str(file_path):
							st.session_state.pop("selected_spectrogram_file", None)
						st.session_state.pop(check_key, None)
						st.rerun()

			if total_pages > 1:
				pagination_cols = st.columns([1, 1, 1, 1])
				with pagination_cols[0]:
					if st.button("⬅ Previous", key="prev_page", use_container_width=True, disabled=current_page == 0):
						st.session_state["file_list_page"] = max(0, current_page - 1)
						st.rerun()

				with pagination_cols[1]:
					st.caption(f"Page {current_page + 1} of {total_pages}")

				with pagination_cols[2]:
					if st.button("Next ➡", key="next_page", use_container_width=True, disabled=current_page >= total_pages - 1):
						st.session_state["file_list_page"] = min(total_pages - 1, current_page + 1)
						st.rerun()

				with pagination_cols[3]:
					st.caption(f"Showing {len(page_files)} of {len(visible_files)} file(s)")

	with panel_col:
		with st.container(border=True):
			st.markdown("### Spectrogram")
			selected_path = next((path for path in project_files if str(path) == str(selected_spectrogram)), None)
			if selected_path is None:
				st.caption("Click Spec on a file to open its spectrogram here.")
			else:
				spec_path = _spectrogram_path(selected_path)
				st.caption(selected_path.name)
				if not spec_path.exists():
					st.warning("No spectrogram found for this file. Click Spec again to generate it.")
				else:
					try:
						result, _ = load_spectrogram(spec_path)
					except Exception as exc:
						st.error(f"Could not load spectrogram for {selected_path.name}: {exc}")
					else:
						times = np.asarray(result.times, dtype=np.float32)
						freqs = np.asarray(result.freqs, dtype=np.float32)
						if result.values.size == 0 or times.size == 0 or freqs.size == 0:
							st.warning("Spectrogram data is empty.")
						else:
							values = np.asarray(result.values, dtype=np.float32)
							max_time_bins = 700
							max_freq_bins = 256
							time_step = max(1, int(np.ceil(values.shape[1] / max_time_bins)))
							freq_step = max(1, int(np.ceil(values.shape[0] / max_freq_bins)))

							if time_step > 1 or freq_step > 1:
								plot_values = values[::freq_step, ::time_step]
								plot_times = times[::time_step]
								plot_freqs = freqs[::freq_step]
								st.caption(f"Interactive preview downsampled: time x{time_step}, frequency x{freq_step}")
							else:
								plot_values = values
								plot_times = times
								plot_freqs = freqs

							fig = go.Figure(data=go.Heatmap(z=plot_values, x=plot_times, y=plot_freqs, colorscale="Gray", zmin=-80.0, zmax=0.0, colorbar={"title": "dB"}))
							fig.update_layout(title=f"{result.kind.upper()} Spectrogram", xaxis_title="Time (s)", yaxis_title="Frequency (Hz)", plot_bgcolor="rgba(16, 41, 53, 1)", paper_bgcolor="rgba(0, 0, 0, 0)", font={"color": "#e6f2f2"}, margin={"l": 60, "r": 20, "t": 40, "b": 50})
							fig.update_xaxes(showgrid=False)
							fig.update_yaxes(showgrid=False)
							st.plotly_chart(fig, width="stretch", config={"scrollZoom": True, "displaylogo": False})

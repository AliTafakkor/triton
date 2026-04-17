from pathlib import Path
from typing import Callable
import csv
import streamlit as st
from triton.core.project import Project, load_file_labels, list_project_files
from triton.core.pipeline_runtime import PIPELINE_ACTIONS, default_step_options
from triton.core.pipeline_matrix import (
	generate_matrix_csv,
	run_matrix_csv,
)


def _matrix_storage_dir(project: Project) -> Path:
    """Return directory where generated matrix CSV files are stored."""
    return project.path / "metadata" / "matrices"


def _pipeline_matrix_dir(project: Project, pipeline_name: str) -> Path:
    """Return matrix CSV directory for a specific pipeline."""
    return _matrix_storage_dir(project) / pipeline_name


def _list_saved_matrix_csvs(project: Project, pipeline_name: str) -> list[Path]:
    """List saved matrix CSVs for a pipeline, newest first."""
    matrix_dir = _pipeline_matrix_dir(project, pipeline_name)
    if not matrix_dir.exists():
        return []
    return sorted((p for p in matrix_dir.glob("*.csv") if p.is_file()), key=lambda p: p.stat().st_mtime, reverse=True)


def _pipeline_matrix_option_defaults(project: Project, pipeline) -> list[tuple[int, str, dict[str, object]]]:
    """Return per-step default+stored options for matrix parameter entry."""
    result: list[tuple[int, str, dict[str, object]]] = []
    for index, step in enumerate(pipeline.steps):
        defaults = default_step_options(step, int(project.sample_rate))
        stored = pipeline.step_options.get(str(index), pipeline.step_options.get(step, {}))
        stored_dict = stored if isinstance(stored, dict) else {}
        merged = {**defaults, **stored_dict}
        result.append((index, step, merged))
    return result

def _hero(project: Project | None) -> None:
    status = "No active project" if project is None else f"Active project: {project.name}"
    st.markdown(
        f"""
        <div style="padding: 32px; border-radius: 28px; background: linear-gradient(135deg, var(--hero-start), var(--hero-end)); border: 1px solid var(--panel-border); box-shadow: 0 30px 60px rgba(2, 10, 16, 0.24), inset 0 1px 0 rgba(255,255,255,0.05); margin-bottom: 22px; position: relative; overflow: hidden;">
          <div style="position: absolute; top: 0; right: 0; width: 300px; height: 300px; background: radial-gradient(circle, rgba(255,159,28,0.08) 0%, transparent 70%); pointer-events: none;"></div>
          <div style="font-size: 12px; letter-spacing: 0.16em; text-transform: uppercase; color: var(--hero-kicker); margin-bottom: 12px; font-weight: 600;">Triton workspace</div>
          <div style="font-size: 44px; font-weight: 800; line-height: 1.05; margin-bottom: 14px; background: linear-gradient(135deg, var(--ink), var(--muted)); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Project-shaped audio workflows</div>
          <div style="max-width: 760px; color: var(--hero-body); font-size: 16px; line-height: 1.7; margin-bottom: 16px; opacity: 0.9;">Start from a project anywhere on disk, keep its audio rules consistent, and let Triton handle import, storage, and processing inside that boundary.</div>
          <div style="display: inline-block; padding: 8px 16px; border-radius: 999px; background: var(--hero-pill-bg); color: var(--hero-pill-text); font-size: 13px; font-weight: 700; border: 1px solid rgba(255,159,28,0.15);">{status}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def _render_project_launcher(
    set_active_project: Callable,
    collect_spectrogram_settings: Callable,
    create_project_fn: Callable,
    register_recent_project_fn: Callable,
    load_recent_projects_fn: Callable,
) -> None:
    st.subheader("Open or create a project")
    st.write(
        "Pick an existing project anywhere on disk or create a new one with the canonical audio settings you want Triton to enforce."
    )

    recent_tab, open_tab, create_tab = st.tabs(["Recent", "Open", "Create"])

    with recent_tab:
        recent_projects = load_recent_projects_fn()
        if not recent_projects:
            st.info("No recent projects yet.")
        else:
            for index, project in enumerate(recent_projects):
                col1, col2 = st.columns([5, 1])
                with col1:
                    st.markdown(f"#### {project['name']}")
                    st.caption(project["path"])
                with col2:
                    if st.button("Open", key=f"recent_project_{index}"):
                        try:
                            set_active_project(Path(project["path"]).expanduser())
                        except Exception as exc:
                            st.error(f"Could not open project: {exc}")
                        else:
                            st.rerun()

    with create_tab:
        with st.form("create_project_form"):
            project_name = st.text_input("Project name", value="my-project")
            project_root = st.text_input("Project folder", value=str(Path.home() / "Projects" / "triton" / "my-project"))
            spec_col1, spec_col2, spec_col3, spec_col4 = st.columns(4)
            with spec_col1:
                sample_rate = st.selectbox("Sample rate (Hz)", options=[8000, 16000, 22050, 24000, 32000, 44100, 48000], index=1, key="create_sr")
            with spec_col2:
                channel_mode = st.selectbox("Channels", options=["mono", "stereo"], key="create_channels")
            with spec_col3:
                bit_depth = st.selectbox("Bit depth", options=[8, 16, 24, 32], index=1, key="create_bit_depth")
            with spec_col4:
                file_format = st.selectbox("File format", options=["wav", "flac", "ogg"], index=0, key="create_file_format")
            with st.expander("Spectrogram defaults", expanded=False):
                spectrogram_settings = collect_spectrogram_settings("create")
            create_submitted = st.form_submit_button("Create project", type="primary")

        if create_submitted:
            project_dir = Path(project_root).expanduser()
            if project_dir.name != project_name and project_name.strip():
                project_dir = project_dir.parent / project_name.strip()

            try:
                project = create_project_fn(
                    project_dir,
                    sample_rate=sample_rate,
                    channel_mode=channel_mode,
                    bit_depth=bit_depth,
                    file_format=file_format,
                    spectrogram_settings=spectrogram_settings,
                )
                st.session_state["active_project"] = project
                register_recent_project_fn(project_dir, project.name)
            except Exception as exc:
                st.error(f"Could not create project: {exc}")
            else:
                st.success(f"Project ready: {project.name}")
                st.rerun()

    with open_tab:
        with st.form("open_project_form"):
            project_root = st.text_input("Existing project folder", value=str(Path.home()))
            open_submitted = st.form_submit_button("Open project", type="primary")

        if open_submitted:
            project_dir = Path(project_root).expanduser()
            try:
                set_active_project(project_dir)
            except Exception as exc:
                st.error(f"Could not open project: {exc}")
            else:
                st.success(f"Opened {project_dir.name}")
                st.rerun()


def _render_matrix_tab(
    project: Project,
    project_files: list[Path],
    load_pipelines: Callable,
    log_event: Callable,
    new_run_id: Callable,
) -> None:
    """Render the Pipeline Matrix tab for parameter sweeps."""
    st.markdown("### Pipeline Matrix")
    st.write(
        "Generate and run parameter sweeps across multiple files. Define parameter combinations, generate a CSV, then execute the matrix."
    )

    pipelines = load_pipelines(project)
    pipeline_names = [p.name for p in pipelines]

    if not pipelines:
        st.info("No pipelines yet. Create one in the Pipelines tab first.")
        return

    gen_col, run_col = st.tabs(["Generate Matrix", "Run Matrix"])

    with gen_col:
        st.markdown("#### Generate Parameter Matrix")
        st.write(
            "Select a pipeline, then define matrix values directly per step option. Triton builds the CSV from those combinations."
        )

        with st.form("generate_matrix_form"):
            selected_pipeline = st.selectbox(
                "Select pipeline",
                options=pipeline_names,
                key="matrix_gen_pipeline",
            )

            st.markdown("##### Parameter specifications")
            st.caption("Enter comma-separated values for any step option you want to sweep.")

            pipeline = next(p for p in pipelines if p.name == selected_pipeline)
            step_blocks = _pipeline_matrix_option_defaults(project, pipeline)
            param_specs: list[str] = []

            for step_index, step_name, options in step_blocks:
                step_label = PIPELINE_ACTIONS.get(step_name, step_name)
                with st.expander(f"Step {step_index + 1}: {step_label} ({step_name})", expanded=False):
                    if not options:
                        st.caption("No configurable options for this step.")
                    for option_name, option_value in options.items():
                        input_key = f"matrix_param_{selected_pipeline}_{step_index}_{option_name}"
                        values_csv = st.text_input(
                            f"{option_name}",
                            value="",
                            key=input_key,
                            help=f"Current value: {option_value}",
                            placeholder="value1,value2,value3",
                        )
                        if values_csv.strip():
                            param_specs.append(f"{step_index}.{option_name}={values_csv}")

            with st.expander("Advanced: manual parameter specs", expanded=False):
                manual_specs_text = st.text_area(
                    "One spec per line (step.option=v1,v2)",
                    value="",
                    key="matrix_manual_specs",
                    placeholder="0.target_peak=0.5,0.8\n2.snr_db=-10,-5,0",
                )
                for line in manual_specs_text.splitlines():
                    clean = line.strip()
                    if clean:
                        param_specs.append(clean)

            st.markdown("##### Files to include")
            
            # File selection mode
            file_selection_mode = st.radio(
                "Select files by:",
                options=["Files", "Label(s)"],
                horizontal=True,
                key="matrix_file_selection_mode"
            )

            selected_files_from_picker = st.multiselect(
                "Select files for matrix",
                options=[p.name for p in project_files],
                default=[p.name for p in project_files[:2]] if project_files else [],
                key="matrix_gen_files",
                help="Used when selection mode is Files.",
            )

            # Always render label picker so options are visible immediately even inside forms.
            all_labels = load_file_labels(project.path)
            available_labels = sorted(set(lbl for lbls in all_labels.values() for lbl in lbls))
            selected_labels = st.multiselect(
                "Select labels for matrix",
                options=available_labels,
                key="matrix_gen_labels",
                help="Used when selection mode is Label(s).",
            )
            if not available_labels:
                st.info("No labels found. Use the File Library tab to add labels to your files.")

            selected_files: list[str] = []
            if file_selection_mode == "Files":
                selected_files = list(selected_files_from_picker)
            else:
                for label in selected_labels:
                    label_files = list_project_files(project.path, label=label)
                    selected_files.extend([p.name for p in label_files])
                # Remove duplicates while preserving order.
                seen: set[str] = set()
                selected_files = [name for name in selected_files if not (name in seen or seen.add(name))]

            default_csv_name = f"{selected_pipeline}.matrix.csv"
            output_csv_name = st.text_input(
                "Output CSV filename",
                value=default_csv_name,
                key="matrix_output_name",
                help="Saved under metadata/matrices/<pipeline>/",
            )

            gen_submitted = st.form_submit_button("Generate Matrix CSV", type="primary")

        if gen_submitted:
            if not selected_pipeline:
                st.error("Select a pipeline.")
            elif not selected_files:
                st.error("Select at least one file.")
            elif not param_specs:
                st.error("Define at least one option sweep value.")
            else:
                try:
                    safe_name = output_csv_name.strip() or default_csv_name
                    if not safe_name.lower().endswith(".csv"):
                        safe_name = f"{safe_name}.csv"
                    output_csv = _pipeline_matrix_dir(project, selected_pipeline) / safe_name
                    
                    with st.spinner(f"Generating matrix CSV with {len(param_specs)} parameter(s) and {len(selected_files)} file(s)..."):
                        total_rows = generate_matrix_csv(
                            project,
                            pipeline,
                            output_csv,
                            param_specs,
                            selected_files,
                        )

                    st.success(f"Generated matrix with {total_rows} row(s).")
                    st.caption(f"Saved to: {output_csv}")

                    if selected_pipeline == st.session_state.get("matrix_run_pipeline"):
                        st.session_state["matrix_run_saved_csv"] = output_csv.name

                    with open(output_csv, "r") as f:
                        csv_data = f.read()
                        st.download_button(
                            label="Download CSV",
                            data=csv_data,
                            file_name=output_csv_name,
                            mime="text/csv",
                        )

                    log_event(
                        project.path,
                        "matrix_csv_generated",
                        {
                            "pipeline": selected_pipeline,
                            "csv": str(output_csv),
                            "total_rows": total_rows,
                            "parameters": len(param_specs),
                            "files": len(selected_files),
                        },
                    )

                except ValueError as e:
                    st.error(f"Invalid parameter spec: {e}")
                except Exception as e:
                    st.error(f"Could not generate matrix: {e}")

    with run_col:
        st.markdown("#### Run Matrix from CSV")
        st.write("Select a saved matrix CSV, upload one, or provide a direct path to execute parameter combinations.")

        with st.form("run_matrix_form"):
            run_pipeline = st.selectbox(
                "Select pipeline",
                options=pipeline_names,
                key="matrix_run_pipeline",
            )

            saved_csvs = _list_saved_matrix_csvs(project, run_pipeline)
            saved_names = [item.name for item in saved_csvs]

            source_mode = st.radio(
                "Matrix source",
                options=["Saved", "Upload", "Path"],
                horizontal=True,
                key="matrix_run_source_mode",
            )

            selected_saved_csv = st.selectbox(
                "Saved matrix CSV",
                options=saved_names,
                key="matrix_run_saved_csv",
                disabled=not saved_names,
                help="CSV files from metadata/matrices/<pipeline>/",
            )
            if source_mode == "Saved" and not saved_names:
                st.info("No saved matrix CSVs for this pipeline yet. Generate one first.")

            csv_upload = st.file_uploader(
                "Upload matrix CSV",
                type=["csv"],
                key="matrix_csv_upload",
                disabled=source_mode != "Upload",
            )

            csv_path_text = st.text_input(
                "Matrix CSV path",
                value="",
                key="matrix_csv_path",
                disabled=source_mode != "Path",
                help="Absolute path or project-relative path.",
            )

            custom_run_id = st.text_input(
                "Run ID (optional, auto-generated if blank)",
                key="matrix_custom_run_id",
            )

            group_finals_by_params = st.checkbox(
                "Group final outputs by parameter set",
                value=False,
                key="matrix_group_finals_by_params",
                help="Creates run folder final_by_params/set_####_* folders with final files for each parameter combination.",
            )

            run_submitted = st.form_submit_button("Execute Matrix", type="primary")

        if run_submitted:
            if not run_pipeline:
                st.error("Select a pipeline.")
            else:
                try:
                    pipeline = next(p for p in pipelines if p.name == run_pipeline)
                    run_id = custom_run_id.strip() if custom_run_id.strip() else new_run_id()

                    matrix_csv_path: Path
                    temp_csv: Path | None = None
                    if source_mode == "Saved":
                        if not saved_names:
                            st.error("No saved matrix CSVs available for this pipeline.")
                            return
                        matrix_csv_path = _pipeline_matrix_dir(project, run_pipeline) / selected_saved_csv
                    elif source_mode == "Upload":
                        if csv_upload is None:
                            st.error("Upload a matrix CSV file.")
                            return
                        temp_csv = project.path / ".matrix_temp.csv"
                        with open(temp_csv, "wb") as f:
                            f.write(csv_upload.getbuffer())
                        matrix_csv_path = temp_csv
                    else:
                        if not csv_path_text.strip():
                            st.error("Provide a matrix CSV path.")
                            return
                        candidate = Path(csv_path_text.strip()).expanduser()
                        matrix_csv_path = candidate if candidate.is_absolute() else (project.path / candidate)
                    matrix_csv_path = matrix_csv_path.resolve()

                    matrix_progress = st.progress(0.0, text=f"Preparing matrix run for {run_pipeline}...")

                    def _update_matrix_progress(current: int, total: int, file_value: str) -> None:
                        if total <= 0:
                            matrix_progress.progress(0.0, text=f"{run_pipeline}: preparing rows...")
                            return
                        display_name = file_value or "<missing file>"
                        matrix_progress.progress(current / total, text=f"{run_pipeline}: row {current}/{total} - {display_name}")

                    with st.spinner(f"Executing matrix on pipeline '{run_pipeline}' with {run_id}..."):
                        successes, errors, base_run_dir = run_matrix_csv(
                            project,
                            pipeline,
                            matrix_csv_path,
                            run_id,
                            collect_finals_by_params=group_finals_by_params,
                            progress_callback=_update_matrix_progress,
                        )
                    matrix_progress.progress(1.0, text=f"Completed matrix run for {run_pipeline}")

                    if temp_csv is not None:
                        temp_csv.unlink(missing_ok=True)

                    if successes:
                        st.success(f"Completed {len(successes)} row(s) successfully.")
                        st.caption(f"Run output folder: {base_run_dir}")
                        if group_finals_by_params:
                            st.caption(f"Grouped finals folder: {base_run_dir / 'final_by_params'}")
                        with st.expander("View successful outputs"):
                            for output in successes:
                                st.caption(str(output))

                    if errors:
                        st.warning(f"{len(errors)} row(s) had errors:")
                        with st.expander("View errors"):
                            for error in errors:
                                st.caption(error)

                    log_event(
                        project.path,
                        "matrix_run_completed",
                        {
                            "pipeline": run_pipeline,
                            "run_id": run_id,
                            "matrix_csv": str(matrix_csv_path),
                            "source_mode": source_mode,
                            "group_finals_by_params": bool(group_finals_by_params),
                            "succeeded": len(successes),
                            "failed": len(errors),
                        },
                    )

                except FileNotFoundError as e:
                    st.error(f"File error: {e}")
                except ValueError as e:
                    st.error(f"CSV format error: {e}")
                except Exception as e:
                    st.error(f"Could not run matrix: {e}")
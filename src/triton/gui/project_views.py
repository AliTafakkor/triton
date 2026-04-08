from pathlib import Path
from typing import Callable
import csv
import streamlit as st
from triton.core.project import Project
from triton.core.pipeline_matrix import (
	generate_matrix_csv,
	parse_parameter_specs,
	run_matrix_csv,
)

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
            "Define parameter combinations and select files to create a matrix CSV that can be executed in batch."
        )

        with st.form("generate_matrix_form"):
            selected_pipeline = st.selectbox(
                "Select pipeline",
                options=pipeline_names,
                key="matrix_gen_pipeline",
            )

            st.markdown("##### Parameter specifications")
            st.caption("Example: 0.target_peak=0.5,0.8 or 0.noise_db=-20,-10")

            param_count = st.number_input(
                "Number of parameter specs",
                min_value=0,
                max_value=10,
                value=1,
                key="matrix_param_count",
            )

            param_specs = []
            for i in range(int(param_count)):
                spec = st.text_input(
                    f"Parameter {i + 1}",
                    placeholder="step.option=v1,v2,v3",
                    key=f"matrix_param_{i}",
                )
                if spec.strip():
                    param_specs.append(spec)

            st.markdown("##### Files to include")
            selected_files = st.multiselect(
                "Select files for matrix",
                options=[p.name for p in project_files],
                default=[p.name for p in project_files[:2]] if project_files else [],
                key="matrix_gen_files",
            )

            output_csv_name = st.text_input(
                "Output CSV filename",
                value="matrix.csv",
                key="matrix_output_name",
            )

            gen_submitted = st.form_submit_button("Generate Matrix CSV", type="primary")

        if gen_submitted:
            if not selected_pipeline:
                st.error("Select a pipeline.")
            elif not selected_files:
                st.error("Select at least one file.")
            elif not param_specs:
                st.error("Define at least one parameter spec.")
            else:
                try:
                    pipeline = next(p for p in pipelines if p.name == selected_pipeline)
                    output_csv = project.path / output_csv_name
                    
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
        st.write("Upload a matrix CSV file and execute it to run the pipeline with all parameter combinations.")

        with st.form("run_matrix_form"):
            run_pipeline = st.selectbox(
                "Select pipeline",
                options=pipeline_names,
                key="matrix_run_pipeline",
            )

            csv_upload = st.file_uploader(
                "Upload matrix CSV",
                type=["csv"],
                key="matrix_csv_upload",
            )

            custom_run_id = st.text_input(
                "Run ID (optional, auto-generated if blank)",
                key="matrix_custom_run_id",
            )

            run_submitted = st.form_submit_button("Execute Matrix", type="primary")

        if run_submitted:
            if not run_pipeline:
                st.error("Select a pipeline.")
            elif csv_upload is None:
                st.error("Upload a matrix CSV file.")
            else:
                try:
                    pipeline = next(p for p in pipelines if p.name == run_pipeline)
                    run_id = custom_run_id.strip() if custom_run_id.strip() else new_run_id()
                    
                    # Save uploaded CSV temporarily
                    temp_csv = project.path / ".matrix_temp.csv"
                    with open(temp_csv, "wb") as f:
                        f.write(csv_upload.getbuffer())

                    with st.spinner(f"Executing matrix on pipeline '{run_pipeline}' with {run_id}..."):
                        successes, errors, base_run_dir = run_matrix_csv(
                            project,
                            pipeline,
                            temp_csv,
                            run_id,
                        )

                    # Clean up temp file
                    temp_csv.unlink(missing_ok=True)

                    if successes:
                        st.success(f"Completed {len(successes)} row(s) successfully.")
                        st.caption(f"Run output folder: {base_run_dir}")
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
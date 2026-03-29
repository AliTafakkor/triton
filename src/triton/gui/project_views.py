from pathlib import Path
from typing import Callable
import streamlit as st
from triton.core.project import Project

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
            sample_rate = st.selectbox("Sample rate", options=[8000, 16000, 22050, 24000, 32000, 44100, 48000], index=1, key="create_sr")
            channel_mode = st.radio("Channel mode", options=["mono", "stereo"], horizontal=True, key="create_channels")
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
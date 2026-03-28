import streamlit as st
from triton.core.project import Project


def _hero(project: Project | None) -> None:
    status = "No active project" if project is None else f"Active project: {project.name}"
    st.markdown(
        f"""
        <div style="padding: 28px; border-radius: 28px; background: linear-gradient(135deg, var(--hero-start), var(--hero-end)); border: 1px solid var(--panel-border); box-shadow: 0 30px 60px rgba(2, 10, 16, 0.24); margin-bottom: 18px;">
          <div style="font-size: 13px; letter-spacing: 0.14em; text-transform: uppercase; color: var(--hero-kicker); margin-bottom: 10px;">Triton workspace</div>
          <div style="font-size: 42px; font-weight: 800; line-height: 1.05; margin-bottom: 12px;">Project-shaped audio workflows</div>
          <div style="max-width: 760px; color: var(--hero-body); font-size: 17px; line-height: 1.6; margin-bottom: 14px;">Start from a project anywhere on disk, keep its audio rules consistent, and let Triton handle import, storage, and processing inside that boundary.</div>
          <div style="display: inline-block; padding: 8px 14px; border-radius: 999px; background: var(--hero-pill-bg); color: var(--hero-pill-text); font-size: 13px; font-weight: 700;">{status}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
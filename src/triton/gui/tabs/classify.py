"""Audio classification tab for the Triton GUI."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from triton.core.project import Project, load_file_labels, log_project_event, set_file_labels


@st.cache_resource(show_spinner="Loading AST classification model...")
def _load_model():
    from triton.classify.ast import load_model
    return load_model()


def render_classify_tab(project: Project, project_files: list[Path]) -> None:
    st.markdown("### Auto-Classify Audio Files")
    st.write(
        "Classify the content of project files using the "
        "[Audio Spectrogram Transformer](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593) "
        "trained on AudioSet's 527 sound categories (speech, music, traffic, nature sounds, etc.)."
    )

    if not project_files:
        st.info("Import some audio files first to classify them.")
        return

    col1, col2 = st.columns([3, 1])
    with col1:
        selected_names = st.multiselect(
            "Files to classify",
            options=[p.name for p in project_files],
            help="Select one or more project files.",
        )
    with col2:
        top_k = int(st.number_input("Top labels", min_value=1, max_value=20, value=5, step=1))

    save_labels = st.checkbox(
        "Save top label to project after classification",
        value=True,
        help="Saves the highest-confidence label to the project label system so it can be used for filtering and batch operations.",
    )

    run = st.button("Classify", type="primary", disabled=not selected_names)

    if not run:
        _render_existing_labels(project, project_files)
        return

    selected_paths = [p for p in project_files if p.name in set(selected_names)]

    try:
        extractor, model = _load_model()
    except Exception as exc:
        st.error(f"Could not load classification model: {exc}")
        st.caption("Make sure `transformers` is installed: add it to pixi.toml pypi-dependencies.")
        return

    from triton.classify.ast import classify_file

    results: list[tuple[Path, object]] = []
    errors: list[str] = []

    with st.spinner(f"Classifying {len(selected_paths)} file(s)..."):
        for file_path in selected_paths:
            try:
                result = classify_file(file_path, extractor=extractor, model=model, top_k=top_k)
                results.append((file_path, result))
                if save_labels and result.labels:
                    set_file_labels(project.path, file_path, [result.labels[0]])
            except Exception as exc:
                errors.append(f"{file_path.name}: {exc}")

    if results:
        log_project_event(
            project.path,
            "classification_completed",
            {
                "files": [p.name for p, _ in results],
                "top_k": top_k,
                "labels_saved": save_labels,
            },
        )

        st.success(f"Classified {len(results)} file(s).")
        st.markdown("#### Results")
        for file_path, result in results:
            with st.container(border=True):
                st.markdown(f"**{file_path.name}**")
                rows = [
                    {"Label": label, "Confidence": f"{score:.1%}"}
                    for label, score in zip(result.labels, result.scores)
                ]
                st.dataframe(rows, use_container_width=True, hide_index=True)
                if save_labels:
                    st.caption(f"Saved label: **{result.labels[0]}**")

    for error in errors:
        st.error(error)

    _render_existing_labels(project, project_files)


def _render_existing_labels(project: Project, project_files: list[Path]) -> None:
    all_labels = load_file_labels(project.path)
    labeled = {stem: labels for stem, labels in all_labels.items() if labels}
    if not labeled:
        return

    st.markdown("#### Existing labels")
    rows = []
    for file_path in project_files:
        labels = labeled.get(file_path.stem)
        if labels:
            rows.append({"File": file_path.name, "Labels": ", ".join(labels)})
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)

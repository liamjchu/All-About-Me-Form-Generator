from __future__ import annotations

import streamlit as st

from ai_backend import generate_all_about_me_profile
from eta import (
    GEN_SECONDS,
    PREP_SECONDS,
    EtaTracker,
    estimate_batch_seconds,
    run_with_heartbeat,
)
from file_inputs import MAX_UPLOAD_BYTES, prepare_upload, profile_stem
from leave_guard import set_generation_leave_guard
from pdf_filler import merge_profiles_pdf

# Prep is quick; most wall time is the local text model call(s).
_PREP_SHARE = 0.15
_MAX_UPLOAD_MB = MAX_UPLOAD_BYTES // (1024 * 1024)

st.set_page_config(page_title="All About Me Profile Generator", page_icon="✨")

st.title("All About Me Profile Generator")
st.write(
    "Upload participant information PDFs and turn them into simpler "
    "All About Me profiles."
)

uploaded_files = st.file_uploader(
    "Upload participant PDFs",
    type=["pdf"],
    accept_multiple_files=True,
    help=(
        f"Each PDF becomes one All About Me profile (max {_MAX_UPLOAD_MB} MB each). "
        "Text must be selectable (scanned image-only PDFs are not supported)."
    ),
)

if "generated_profiles" not in st.session_state:
    st.session_state.generated_profiles = []

# Always start disabled — popups only while generation is actually running.
st.session_state["generation_in_progress"] = False
set_generation_leave_guard(False)


if st.button("Generate Profiles", type="primary", use_container_width=True):
    if not uploaded_files:
        st.warning("Upload at least one PDF first.")
        st.session_state.generated_profiles = []
    else:
        st.session_state["generation_in_progress"] = True
        # flush=True: wait so the browser installs parent-page confirm handlers
        # before this script blocks on local AI generation.
        set_generation_leave_guard(True, flush=True)
        st.info(
            "Generation in progress — changing uploads or leaving this page "
            "will ask you to confirm first."
        )
        generated: list[dict] = []
        try:
            initial_seconds = estimate_batch_seconds(file_count=len(uploaded_files))
            eta = EtaTracker(total_units=initial_seconds)
            eta_box = st.empty()
            progress = st.progress(0, text="Preparing uploads…")

            def refresh_progress(fraction: float, status: str) -> None:
                eta_box.info(eta.label(status))
                progress.progress(min(max(fraction, 0.0), 1.0), text=status)

            refresh_progress(0.0, "Preparing uploads…")

            prepared_list: list[tuple[str, str]] = []
            total_files = len(uploaded_files)
            for index, uploaded_file in enumerate(uploaded_files):
                status = f"Reading {uploaded_file.name}…"
                refresh_progress(_PREP_SHARE * (index / max(total_files, 1)), status)
                try:
                    prepared = prepare_upload(
                        file_name=uploaded_file.name,
                        file_bytes=uploaded_file.getvalue(),
                        mime_type=uploaded_file.type,
                    )
                except ValueError as error:
                    st.error(f"Could not read {uploaded_file.name}: {error}")
                    continue

                prepared_list.append((uploaded_file.name, prepared.raw_text))
                eta.add_completed(PREP_SECONDS)
                refresh_progress(
                    _PREP_SHARE * ((index + 1) / max(total_files, 1)),
                    status,
                )

            profile_total = max(len(prepared_list), 1)
            eta.set_total(eta.completed_units + len(prepared_list) * GEN_SECONDS)

            gen_span = 1.0 - _PREP_SHARE
            for profile_index, (file_name, raw_text) in enumerate(prepared_list):
                label = profile_stem(file_name)
                status_prefix = f"Creating profile for {label}"
                base = _PREP_SHARE + gen_span * (profile_index / profile_total)
                slice_span = gen_span / profile_total

                def on_tick(elapsed: float, within: float) -> None:
                    eta.set_provisional(min(elapsed, GEN_SECONDS * 0.98))
                    refresh_progress(
                        base + slice_span * within,
                        f"{status_prefix} ({int(elapsed)}s — local AI reading PDF…)",
                    )

                try:
                    markdown, pdf_bytes = run_with_heartbeat(
                        lambda raw_text=raw_text: generate_all_about_me_profile(
                            raw_text=raw_text
                        ),
                        on_tick=on_tick,
                        expected_seconds=GEN_SECONDS,
                        poll_seconds=0.8,
                    )
                except (RuntimeError, ValueError) as error:
                    st.error(f"Could not create profile from {file_name}: {error}")
                    eta.add_completed(GEN_SECONDS)
                    continue

                generated.append(
                    {
                        "stem": label,
                        "markdown": markdown,
                        "pdf_bytes": pdf_bytes,
                        "sources": [file_name],
                    }
                )
                eta.add_completed(GEN_SECONDS)
                refresh_progress(
                    _PREP_SHARE + gen_span * ((profile_index + 1) / profile_total),
                    f"{status_prefix} — done",
                )

            eta_box.success("Ready — you can download your files below.")
            progress.progress(1.0, text="Done.")
            st.session_state.generated_profiles = generated
        finally:
            st.session_state["generation_in_progress"] = False
            set_generation_leave_guard(False)

profiles = st.session_state.generated_profiles
if profiles:
    st.success(f"Created {len(profiles)} profile{'s' if len(profiles) != 1 else ''}.")
    for profile in profiles:
        with st.expander(profile["stem"], expanded=False):
            st.markdown(profile["markdown"])

    merged_pdf = merge_profiles_pdf(profiles)
    downloaded = st.download_button(
        "Download all profiles",
        data=merged_pdf,
        file_name="all-about-me-profiles.pdf",
        mime="application/pdf",
        type="primary",
        use_container_width=True,
        key="download-all-profiles",
    )
    if downloaded:
        # Download payload was already sent; clear session PII on this rerun.
        st.session_state.generated_profiles = []
        st.rerun()

    if st.button("Wipe profiles from this session", use_container_width=True):
        st.session_state.generated_profiles = []
        st.rerun()

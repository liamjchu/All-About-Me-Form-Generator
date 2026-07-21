from __future__ import annotations

from pathlib import Path

import streamlit as st

from ai_backend import generate_all_about_me_profile
from file_inputs import prepare_upload
from pdf_filler import merge_profiles_pdf

st.set_page_config(page_title="All About Me Profile Generator", page_icon="✨")

st.title("All About Me Profile Generator")
st.write(
    "Upload participant information files and turn them into simpler "
    "All About Me profiles."
)

uploaded_files = st.file_uploader(
    "Upload participant information",
    type=["txt", "csv", "pdf", "png", "jpg", "jpeg"],
    accept_multiple_files=True,
    help="You can select one or more text, CSV, PDF, or image files.",
)

if "generated_profiles" not in st.session_state:
    st.session_state.generated_profiles = []


if st.button("Generate Profiles", type="primary", use_container_width=True):
    if not uploaded_files:
        st.warning("Upload at least one text, CSV, PDF, or image file first.")
        st.session_state.generated_profiles = []
    else:
        generated: list[dict] = []
        progress = st.progress(0, text="Starting…")
        total = len(uploaded_files)

        for index, uploaded_file in enumerate(uploaded_files):
            progress.progress(
                index / total,
                text=f"Creating profile for {uploaded_file.name}…",
            )
            try:
                prepared = prepare_upload(
                    file_name=uploaded_file.name,
                    file_bytes=uploaded_file.getvalue(),
                    mime_type=uploaded_file.type,
                )
                markdown, pdf_bytes = generate_all_about_me_profile(
                    raw_text=prepared.raw_text,
                    image_bytes=prepared.image_bytes,
                    image_mime_type=prepared.image_mime_type,
                )
            except (RuntimeError, ValueError) as error:
                st.error(f"Could not create {uploaded_file.name}: {error}")
                continue

            stem = Path(uploaded_file.name).stem
            generated.append(
                {
                    "stem": stem,
                    "markdown": markdown,
                    "pdf_bytes": pdf_bytes,
                }
            )

        progress.progress(1.0, text="Done.")
        st.session_state.generated_profiles = generated

profiles = st.session_state.generated_profiles
if profiles:
    st.success(f"Created {len(profiles)} profile{'s' if len(profiles) != 1 else ''}.")
    for profile in profiles:
        with st.expander(profile["stem"], expanded=False):
            st.markdown(profile["markdown"])

    merged_pdf = merge_profiles_pdf(profiles)
    st.download_button(
        "Download all profiles",
        data=merged_pdf,
        file_name="all-about-me-profiles.pdf",
        mime="application/pdf",
        type="primary",
        use_container_width=True,
        key="download-all-profiles",
    )

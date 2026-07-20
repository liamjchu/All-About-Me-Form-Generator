from pathlib import Path

import streamlit as st

from ai_backend import generate_all_about_me_profile
from file_inputs import prepare_upload

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

if st.button("Generate Profiles", type="primary", use_container_width=True):
    if not uploaded_files:
        st.warning("Upload at least one text, CSV, PDF, or image file first.")

    for uploaded_file in uploaded_files or []:
        with st.spinner(
            f"Extracting text and generating profile for {uploaded_file.name}..."
        ):
            try:
                # 1) Extract usable text/image content from the upload.
                prepared = prepare_upload(
                    file_name=uploaded_file.name,
                    file_bytes=uploaded_file.getvalue(),
                    mime_type=uploaded_file.type,
                )
                # 2) Pass that content through the local Ollama pipeline.
                markdown, pdf_bytes = generate_all_about_me_profile(
                    raw_text=prepared.raw_text,
                    image_bytes=prepared.image_bytes,
                    image_mime_type=prepared.image_mime_type,
                )
            except (RuntimeError, ValueError) as error:
                st.error(f"Could not create {uploaded_file.name}: {error}")
                continue

        stem = Path(uploaded_file.name).stem
        st.subheader(stem)
        # 3) Show the formatted Markdown profile on screen.
        st.markdown(markdown)
        st.download_button(
            "Download PDF",
            data=pdf_bytes,
            file_name=f"{stem}-all-about-me.pdf",
            mime="application/pdf",
            key=f"download-{uploaded_file.name}",
        )

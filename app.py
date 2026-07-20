from pathlib import Path

import streamlit as st

from ai_backend import generate_all_about_me_markdown

st.set_page_config(page_title="All About Me Profile Generator", page_icon="✨")

st.title("All About Me Profile Generator")
st.write(
    "Upload participant information files and turn them into simpler "
    "All About Me profiles."
)

uploaded_files = st.file_uploader(
    "Upload participant information",
    type=["txt", "csv", "png", "jpg", "jpeg"],
    accept_multiple_files=True,
    help="You can select one or more text, CSV, or image files.",
)

if st.button("Generate Profiles", type="primary", use_container_width=True):
    if not uploaded_files:
        st.warning("Upload at least one text, CSV, or image file first.")

    for uploaded_file in uploaded_files or []:
        with st.spinner(f"Creating {uploaded_file.name}..."):
            try:
                file_bytes = uploaded_file.getvalue()
                if uploaded_file.type.startswith("image/"):
                    profile = generate_all_about_me_markdown(
                        image_bytes=file_bytes,
                        image_mime_type=uploaded_file.type,
                    )
                else:
                    profile = generate_all_about_me_markdown(
                        raw_text=file_bytes.decode("utf-8", errors="replace")
                    )
            except (RuntimeError, ValueError) as error:
                st.error(f"Could not create {uploaded_file.name}: {error}")
                continue

        st.subheader(Path(uploaded_file.name).stem)
        st.markdown(profile)
        st.download_button(
            "Download Markdown",
            data=profile,
            file_name=f"{Path(uploaded_file.name).stem}-all-about-me.md",
            mime="text/markdown",
            key=f"download-{uploaded_file.name}",
        )

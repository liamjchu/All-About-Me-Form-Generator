import streamlit as st

st.set_page_config(page_title="All About Me Profile Generator", page_icon="✨")

st.title("All About Me Profile Generator")
st.write(
    "Upload participant information files and turn them into visually appealing "
    "All About Me profiles."
)

uploaded_files = st.file_uploader(
    "Upload participant information",
    type=["txt", "csv"],
    accept_multiple_files=True,
    help="You can select one or more .txt or .csv files.",
)

st.button("**Generate Profiles**", type="primary", use_container_width=True)

from __future__ import annotations

import streamlit as st

from ai_backend import extract_footer_mark, generate_all_about_me_profile
from eta import (
    GEN_TEXT_SECONDS,
    GEN_VISION_SECONDS,
    PREP_IMAGE_SECONDS,
    PREP_TEXT_SECONDS,
    EtaTracker,
    estimate_batch_seconds,
    run_with_heartbeat,
)
from file_inputs import (
    FooterMark,
    crop_footer_band,
    group_label,
    group_upload_indices,
    prepare_upload,
)
from pdf_filler import merge_profiles_pdf

# Prep is quick; most wall time is the local vision model call(s).
_PREP_SHARE = 0.15

st.set_page_config(page_title="All About Me Profile Generator", page_icon="✨")

st.title("All About Me Profile Generator")
st.write(
    "Upload participant information files and turn them into simpler "
    "All About Me profiles."
)

with st.expander("How multi-page photo uploads work", expanded=False):
    st.markdown(
        """
Upload photos **in order**. Pages that belong to the same participant should sit
next to each other in the upload list (for example Bob Joe’s page 45, then page 46).

After each photo is oriented upright, the app reads the printed footer:
- **Bottom right:** page marks like `45 of 85` and `46 of 85`
- **Bottom left:** the matching timestamp on that same line
  (for example `Jun 30 2026 1:21PM ET`)

Consecutive photos are combined into **one** All About Me profile when those
footer cues say they are pages of the same form. Text / CSV / PDF uploads stay
one file → one profile.

**Tip:** Photograph each page clearly — fill the frame, keep it sharp and well
lit, and avoid glare or blur so the printed text (including the footer) is easy
to read.

**Progress tip:** After photos are prepared, the bar may sit for a while on
“Creating profile…” while the local vision model reads the pages. That step is
working — it often takes about a minute per participant on a laptop.
"""
    )


uploaded_files = st.file_uploader(
    "Upload participant information",
    type=["txt", "csv", "pdf", "png", "jpg", "jpeg"],
    accept_multiple_files=True,
    help=(
        "For phone photos of a multi-page form, keep that person's pages next "
        "to each other. Footer page numbers and matching timestamps are used "
        "to group them."
    ),
)

pages_per_form = st.number_input(
    "Pages per participant form",
    min_value=1,
    max_value=6,
    value=2,
    help="How many consecutive photo pages make one intake form (usually 2).",
)

if "generated_profiles" not in st.session_state:
    st.session_state.generated_profiles = []


if st.button("Generate Profiles", type="primary", use_container_width=True):
    if not uploaded_files:
        st.warning("Upload at least one text, CSV, PDF, or image file first.")
        st.session_state.generated_profiles = []
    else:
        generated: list[dict] = []
        file_names = [uploaded.name for uploaded in uploaded_files]
        mime_types = [uploaded.type for uploaded in uploaded_files]
        initial_seconds = estimate_batch_seconds(
            file_names=file_names,
            mime_types=mime_types,
            pages_per_form=int(pages_per_form),
        )
        eta = EtaTracker(total_units=initial_seconds)
        eta_box = st.empty()
        progress = st.progress(0, text="Preparing uploads…")

        def refresh_progress(fraction: float, status: str) -> None:
            eta_box.info(eta.label(status))
            progress.progress(min(max(fraction, 0.0), 1.0), text=status)

        refresh_progress(0.0, "Preparing uploads…")

        prepared_list = []
        names: list[str] = []
        is_image: list[bool] = []
        footers: list[FooterMark | None] = []

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

            names.append(uploaded_file.name)
            prepared_list.append(prepared)
            if prepared.image_bytes:
                is_image.append(True)
                try:
                    footer_bytes, footer_mime = crop_footer_band(
                        prepared.image_bytes,
                        mime_type=prepared.image_mime_type,
                    )
                    footers.append(
                        extract_footer_mark(
                            footer_bytes,
                            image_mime_type=footer_mime,
                        )
                    )
                except (RuntimeError, ValueError) as error:
                    st.warning(
                        f"Could not read footer on {uploaded_file.name}: {error}. "
                        "That page will not auto-group."
                    )
                    footers.append(FooterMark())
                eta.add_completed(PREP_IMAGE_SECONDS)
            else:
                is_image.append(False)
                footers.append(None)
                eta.add_completed(PREP_TEXT_SECONDS)

            refresh_progress(
                _PREP_SHARE * ((index + 1) / max(total_files, 1)),
                status,
            )

        groups = group_upload_indices(
            is_image=is_image,
            footers=footers,
            pages_per_form=int(pages_per_form),
        )
        group_total = max(len(groups), 1)
        vision_groups = 0
        text_groups = 0
        for indices in groups:
            if any(prepared_list[i].image_bytes for i in indices):
                vision_groups += 1
            else:
                text_groups += 1
        eta.set_total(
            eta.completed_units
            + vision_groups * GEN_VISION_SECONDS
            + text_groups * GEN_TEXT_SECONDS
        )

        gen_span = 1.0 - _PREP_SHARE
        for group_index, indices in enumerate(groups):
            group_names = [names[i] for i in indices]
            group_prepared = [prepared_list[i] for i in indices]
            group_footers = [footers[i] for i in indices]
            label = group_label(file_names=group_names, footers=group_footers)
            status_prefix = f"Creating profile for {label}"
            base = _PREP_SHARE + gen_span * (group_index / group_total)
            slice_span = gen_span / group_total

            text_parts = [
                part.raw_text.strip()
                for part in group_prepared
                if part.raw_text and part.raw_text.strip()
            ]
            images = [
                (part.image_bytes, part.image_mime_type)
                for part in group_prepared
                if part.image_bytes
            ]
            expected = GEN_VISION_SECONDS if images else GEN_TEXT_SECONDS
            joined_text = "\n\n".join(text_parts) if text_parts else None
            image_list = images or None

            def on_tick(elapsed: float, within: float) -> None:
                # Credit wall time into the current slice so ETA counts down
                # while the model runs (capped just shy of full until done).
                eta.set_provisional(min(elapsed, expected * 0.98))
                refresh_progress(
                    base + slice_span * within,
                    f"{status_prefix} ({int(elapsed)}s — local AI reading photos…)",
                )

            try:
                markdown, pdf_bytes = run_with_heartbeat(
                    lambda joined_text=joined_text, image_list=image_list: (
                        generate_all_about_me_profile(
                            raw_text=joined_text,
                            images=image_list,
                        )
                    ),
                    on_tick=on_tick,
                    expected_seconds=expected,
                    poll_seconds=0.8,
                )
            except (RuntimeError, ValueError) as error:
                joined = ", ".join(group_names)
                st.error(f"Could not create profile from {joined}: {error}")
                # Still count planned work so the ETA does not stall forever.
                eta.add_completed(expected)
                continue

            generated.append(
                {
                    "stem": label,
                    "markdown": markdown,
                    "pdf_bytes": pdf_bytes,
                    "sources": group_names,
                }
            )
            eta.add_completed(expected)
            refresh_progress(
                _PREP_SHARE + gen_span * ((group_index + 1) / group_total),
                f"{status_prefix} — done",
            )

        eta_box.success("Ready — you can download your files below.")
        progress.progress(1.0, text="Done.")
        st.session_state.generated_profiles = generated

profiles = st.session_state.generated_profiles
if profiles:
    st.success(f"Created {len(profiles)} profile{'s' if len(profiles) != 1 else ''}.")
    for profile in profiles:
        sources = profile.get("sources") or []
        with st.expander(profile["stem"], expanded=False):
            if len(sources) > 1:
                st.caption("Grouped from: " + ", ".join(sources))
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

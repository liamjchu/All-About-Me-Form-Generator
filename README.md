# All About Me Form Generator

Convert participant information into clean, simple All About Me profiles.

## Setup

1. Create a virtual environment, then install the pinned dependencies:
   ```powershell
   py -m pip install -r requirements.txt
   ```
2. Add your API key to the blank `.env` file in the project root:
   ```text
   OPENAI_API_KEY=your_api_key_here
   ```
   `.env` is ignored by Git and must not be shared.
3. Start the app:
   ```powershell
   streamlit run app.py
   ```

The app sends one selected text document or image at a time to `gpt-4o-mini`.
Images are read directly by the model; the output preserves the Markdown layout
in `all_about_me_template.md`. Edit that template to change the profile fields.

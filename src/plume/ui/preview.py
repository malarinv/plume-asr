from pathlib import Path

import streamlit as st
import typer
from plume.utils import ExtendedPath
from plume.utils.ui_persist import setup_file_state

app = typer.Typer()

setup_file_state(st)


@st.cache()
def load_ui_data(validation_ui_data_path: Path):
    typer.echo(f"Using validation ui data from {validation_ui_data_path}")
    return list(ExtendedPath(validation_ui_data_path).read_jsonl())


@app.command()
def main(manifest: Path):
    asr_data = load_ui_data(manifest)
    sample_no = st.get_current_cursor()
    if len(asr_data) - 1 < sample_no or sample_no < 0:
        print("Invalid samplno resetting to 0")
        st.update_cursor(0)
    sample = asr_data[sample_no]
    st.title("ASR Manifest Preview")
    st.markdown(f"{sample_no+1} of {len(asr_data)} : **{sample['text']}**")
    new_sample = st.number_input(
        "Go To Sample:",
        value=sample_no + 1,
        min_value=1,
        max_value=len(asr_data),
    )
    if new_sample != sample_no + 1:
        st.update_cursor(new_sample - 1)
    st.sidebar.markdown(f"Gold Text: **{sample['text']}**")
    st.audio((manifest.parent / Path(sample["audio_filepath"])).open("rb"))


if __name__ == "__main__":
    try:
        app()
    except SystemExit:
        pass

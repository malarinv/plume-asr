# import sys
from pathlib import Path

import streamlit as st
import typer
from plume.utils import ExtendedPath
from plume.utils.ui_persist import setup_mongo_asr_validation_state

app = typer.Typer()

setup_mongo_asr_validation_state(st)


@st.cache()
def load_ui_data(data_dir: Path, dump_fname: Path):
    annotation_ui_data_path = data_dir / dump_fname
    typer.echo(f"Using annotation ui data from {annotation_ui_data_path}")
    return ExtendedPath(annotation_ui_data_path).read_json()


def show_key(sample, key, trail=""):
    if key in sample:
        title = key.replace("_", " ").title()
        if type(sample[key]) == float:
            st.sidebar.markdown(f"{title}: {sample[key]:.2f}{trail}")
        else:
            st.sidebar.markdown(f"{title}: {sample[key]}")


@app.command()
def main(data_dir: Path, dump_fname: Path = "ui_dump.json", task_id: str = ""):
    st.set_task(data_dir, task_id)
    ui_config = load_ui_data(data_dir, dump_fname)
    asr_data = ui_config["data"]
    annotation_only = ui_config.get("annotation_only", False)
    asr_result_key = ui_config.get("asr_result_key", "pretrained_asr")
    sample_no = st.get_current_cursor()
    if len(asr_data) - 1 < sample_no or sample_no < 0:
        print("Invalid samplno resetting to 0")
        st.update_cursor(0)
    sample = asr_data[sample_no]
    task_uid = st.task_id.rsplit("-", 1)[1]
    if annotation_only:
        st.title(f"ASR Annotation - # {task_uid}")
    else:
        st.title(f"ASR Validation - # {task_uid}")
    st.markdown(f"{sample_no+1} of {len(asr_data)} : **{sample['text']}**")
    new_sample = st.number_input(
        "Go To Sample:", value=sample_no + 1, min_value=1, max_value=len(asr_data)
    )
    if new_sample != sample_no + 1:
        st.update_cursor(new_sample - 1)
    st.sidebar.title(f"Details: [{sample['real_idx']}]")
    st.sidebar.markdown(f"Gold Text: **{sample['text']}**")
    # if "caller" in sample:
    #     st.sidebar.markdown(f"Caller: **{sample['caller']}**")
    show_key(sample, "caller")
    if not annotation_only:
        show_key(sample, asr_result_key)
        show_key(sample, "asr_wer", trail="%")
        show_key(sample, "correct_candidate")

    st.sidebar.image((data_dir / Path(sample["plot_path"])).read_bytes())
    st.audio((data_dir / Path(sample["audio_path"])).open("rb"))
    # set default to text
    corrected = sample["text"]
    correction_entry = st.get_correction_entry(sample["utterance_id"])
    selected_idx = 0
    options = ("Correct", "Incorrect", "Inaudible")
    # if correction entry is present set the corresponding ui defaults
    if correction_entry:
        selected_idx = options.index(correction_entry["value"]["status"])
        corrected = correction_entry["value"]["correction"]
    selected = st.radio("The Audio is", options, index=selected_idx)
    if selected == "Incorrect":
        corrected = st.text_input("Actual:", value=corrected)
    if selected == "Inaudible":
        corrected = ""
    if st.button("Submit"):
        st.update_entry(
            sample["utterance_id"], {"status": selected, "correction": corrected}
        )
        st.update_cursor(sample_no + 1)
    if correction_entry:
        status = correction_entry["value"]["status"]
        correction = correction_entry["value"]["correction"]
        st.markdown(f"Your Response: **{status}** Correction: **{correction}**")
    text_sample = st.text_input("Go to Text:", value="")
    if text_sample != "":
        candidates = [i for (i, p) in enumerate(asr_data) if p["text"] == text_sample]
        if len(candidates) > 0:
            st.update_cursor(candidates[0])
    real_idx = st.number_input(
        "Go to real-index",
        value=sample["real_idx"],
        min_value=0,
        max_value=len(asr_data) - 1,
    )
    if real_idx != int(sample["real_idx"]):
        idx = [i for (i, p) in enumerate(asr_data) if p["real_idx"] == real_idx][0]
        st.update_cursor(idx)


if __name__ == "__main__":
    try:
        app()
    except SystemExit:
        pass

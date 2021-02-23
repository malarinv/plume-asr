# import sys
from pathlib import Path
from uuid import uuid4

import streamlit as st
import typer

from plume.utils import ExtendedPath, get_mongo_conn
from plume.preview.st_rerun import rerun

app = typer.Typer()


if not hasattr(st, "mongo_connected"):
    st.mongoclient = get_mongo_conn(col="asr_validation")
    mongo_conn = st.mongoclient
    st.task_id = str(uuid4())

    def current_cursor_fn():
        # mongo_conn = st.mongoclient
        cursor_obj = mongo_conn.find_one(
            {"type": "current_cursor", "task_id": st.task_id}
        )
        cursor_val = cursor_obj["cursor"]
        return cursor_val

    def update_cursor_fn(val=0):
        mongo_conn.find_one_and_update(
            {"type": "current_cursor", "task_id": st.task_id},
            {"$set": {"type": "current_cursor", "task_id": st.task_id, "cursor": val}},
            upsert=True,
        )
        rerun()

    def get_correction_entry_fn(code):
        return mongo_conn.find_one(
            {"type": "correction", "code": code}, projection={"_id": False}
        )

    def update_entry_fn(code, value):
        mongo_conn.find_one_and_update(
            {"type": "correction", "code": code},
            {"$set": {"value": value, "task_id": st.task_id}},
            upsert=True,
        )

    def set_task_fn(data_path, task_id):
        if task_id:
            st.task_id = task_id
        task_path = data_path / Path(f"task-{st.task_id}.lck")
        if not task_path.exists():
            print(f"creating task lock at {task_path}")
            task_path.touch()

    st.get_current_cursor = current_cursor_fn
    st.update_cursor = update_cursor_fn
    st.get_correction_entry = get_correction_entry_fn
    st.update_entry = update_entry_fn
    st.set_task = set_task_fn
    st.mongo_connected = True
    cursor_obj = mongo_conn.find_one({"type": "current_cursor", "task_id": st.task_id})
    if not cursor_obj:
        update_cursor_fn(0)


@st.cache()
def load_ui_data(data_dir: Path, dump_fname: Path):
    validation_ui_data_path = data_dir / dump_fname
    typer.echo(f"Using validation ui data from {validation_ui_data_path}")
    return ExtendedPath(validation_ui_data_path).read_json()


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

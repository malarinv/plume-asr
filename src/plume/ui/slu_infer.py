# from pathlib import Path

import streamlit as st
import typer

from plume.utils.transcribe import triton_transcribe_grpc_gen
from plume.utils.audio import audio_wav_bytes_to_seg

app = typer.Typer()

transcriber, prep = triton_transcribe_grpc_gen(
    asr_model="slu_num_wav2vec2", method="whole", append_raw=True
)


@app.command()
def main():
    st.title("SLU Inference")
    audio_file = st.file_uploader("Upload File", type=["wav", "mp3"])
    if audio_file:
        audio_bytes = audio_file.read()
        seg = audio_wav_bytes_to_seg(audio_bytes)
        st.audio(audio_bytes)
        tscript = transcriber(prep(seg))
        st.write(tscript)


if __name__ == "__main__":
    try:
        app()
    except SystemExit:
        pass

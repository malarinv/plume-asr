from pathlib import Path

import streamlit as st
import typer

app = typer.Typer()


@app.command()
def main(wav_dir: Path):
    wav_file = list(wav_dir.glob('**/*.wav'))[0]
    st.title("Audio Preview")
    print(wav_file.exists())
    st.audio(str(wav_dir / wav_file))


if __name__ == "__main__":
    try:
        app()
    except SystemExit:
        pass

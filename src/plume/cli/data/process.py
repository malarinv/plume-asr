from pathlib import Path

import typer
from ...utils.audio import remove_if_invalid, copy_channel_to
import shutil

app = typer.Typer()


@app.callback()
def process():
    """
    clean sub commands
    """


@app.command()
def remove_invalid(audio_dir: Path, out_dir: Path):
    shutil.copytree(audio_dir, out_dir, dirs_exist_ok=True)
    aud_files = list(out_dir.glob("*.mp3")) + list(out_dir.glob("*.wav"))
    for af in aud_files:
        remove_if_invalid(af)


@app.command()
def extract_channel(audio_dir: Path, out_dir: Path, channel="left"):
    # shutil.copytree(audio_dir, out_dir, dirs_exist_ok=True)
    out_dir.mkdir(exist_ok=True, parents=True)
    aud_files = list(audio_dir.glob("*.mp3")) + list(audio_dir.glob("*.wav"))
    for af in aud_files:
        out_af = out_dir / af.relative_to(audio_dir)
        copy_channel_to(af, out_af, channel)

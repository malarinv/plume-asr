from pathlib import Path

import typer
from ...utils.tts import GoogleTTS

app = typer.Typer()


@app.command()
def tts_dataset(dest_path: Path):
    tts = GoogleTTS()
    pass

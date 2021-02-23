import typer
from ..models.wav2vec2.eval import app as wav2vec2_app

app = typer.Typer()
app.add_typer(wav2vec2_app, name="wav2vec2")

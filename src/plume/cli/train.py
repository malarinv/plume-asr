import typer
from ..models.wav2vec2.train import app as wav2vec2_app

app = typer.Typer()
app.add_typer(wav2vec2_app, name="wav2vec2")


@app.callback()
def train():
    """
    train sub commands
    """

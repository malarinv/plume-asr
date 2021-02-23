import typer
from ..models.wav2vec2.train import app as train_app

app = typer.Typer()
app.add_typer(train_app, name="wav2vec2")

import typer
from ..models.wav2vec2.eval import app as wav2vec2_app
from ..models.wav2vec2_transformers.eval import app as wav2vec2_transformers_app

app = typer.Typer()
app.add_typer(wav2vec2_app, name="wav2vec2")
app.add_typer(wav2vec2_transformers_app, name="wav2vec2_transformers")


@app.callback()
def eval():
    """
    eval sub commands
    """

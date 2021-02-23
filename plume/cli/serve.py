import typer
from ..models.wav2vec2.serve import app as wav2vec2_app
from ..models.jasper.serve import app as jasper_app

app = typer.Typer()
app.add_typer(wav2vec2_app, name="wav2vec2")
app.add_typer(jasper_app, name="jasper")

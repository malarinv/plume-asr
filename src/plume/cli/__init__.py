import typer
from ..utils import app as utils_app
from .data import app as data_app
from ..ui import app as ui_app
from .train import app as train_app
from .eval import app as eval_app
from .serve import app as serve_app

app = typer.Typer()
app.add_typer(data_app)
app.add_typer(ui_app)
app.add_typer(train_app)
app.add_typer(eval_app)
app.add_typer(serve_app)
app.add_typer(utils_app)


def main():
    app()


if __name__ == "__main__":
    main()

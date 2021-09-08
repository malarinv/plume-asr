import typer

# from fairseq_cli.train import cli_main
# import sys
from pathlib import Path
# import shlex
from plume.utils import lazy_callable

cli_main = lazy_callable("fairseq_cli.train.cli_main")

app = typer.Typer()


@app.command()
def local(dataset_path: Path):
    pass


if __name__ == "__main__":
    cli_main()

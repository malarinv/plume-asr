import typer
import sys
from pathlib import Path

from plume.utils import lazy_module

# from streamlit import cli as stcli

stcli = lazy_module("streamlit.cli")
app = typer.Typer()


@app.callback()
def ui():
    """
    ui sub commands
    """


@app.command()
def annotation(data_dir: Path, dump_fname: Path = "ui_dump.json", task_id: str = ""):
    annotation_lit_path = Path(__file__).parent / Path("annotation.py")
    if task_id:
        sys.argv = [
            "streamlit",
            "run",
            str(annotation_lit_path),
            "--",
            str(data_dir),
            "--task-id",
            task_id,
            "--dump-fname",
            dump_fname,
        ]
    else:
        sys.argv = [
            "streamlit",
            "run",
            str(annotation_lit_path),
            "--",
            str(data_dir),
            "--dump-fname",
            dump_fname,
        ]
    sys.exit(stcli.main())


@app.command()
def preview(manifest_path: Path, port: int = 8081):
    annotation_lit_path = Path(__file__).parent / Path("preview.py")
    sys.argv = [
        "streamlit",
        "run",
        "--server.port",
        str(port),
        str(annotation_lit_path),
        "--",
        str(manifest_path),
    ]
    sys.exit(stcli.main())


@app.command()
def encrypted_preview(manifest_path: Path, key: str, port: int = 8081):
    lit_path = Path(__file__).parent / Path("encrypted_preview.py")
    sys.argv = [
        "streamlit",
        "run",
        "--server.port",
        str(port),
        str(lit_path),
        "--",
        str(manifest_path),
        str(key),
    ]
    sys.exit(stcli.main())


@app.command()
def audio(audio_dir: Path):
    lit_path = Path(__file__).parent / Path("audio.py")
    sys.argv = ["streamlit", "run", str(lit_path), "--", str(audio_dir)]
    sys.exit(stcli.main())


@app.command()
def collection(data_dir: Path, task_id: str = ""):
    # TODO: Implement web ui for data collection
    pass


@app.command()
def alignment(preview_dir: Path, port: int = 8010):
    from RangeHTTPServer import RangeRequestHandler
    from functools import partial
    from http.server import HTTPServer

    server_address = ("", port)
    handler_class = partial(RangeRequestHandler, directory=str(preview_dir))
    httpd = HTTPServer(server_address, handler_class)
    httpd.serve_forever()


def main():
    app()


if __name__ == "__main__":
    main()

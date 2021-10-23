import os
# import logging
from pathlib import Path

# from rpyc.utils.server import ThreadedServer
import typer

from ...utils.serve import ASRService
from plume.utils import lazy_callable

# from plume.models.wav2vec2_transformers.asr import Wav2Vec2TransformersASR
# from .asr import Wav2Vec2ASR
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
# )

ThreadedServer = lazy_callable("rpyc.utils.server.ThreadedServer")
Wav2Vec2TransformersASR = lazy_callable(
    "plume.models.wav2vec2_transformers.asr.Wav2Vec2TransformersASR"
)

app = typer.Typer()


# @app.command()
# def rpyc(
#     w2v_path: Path = "/path/to/base.pt",
#     ctc_path: Path = "/path/to/ctc.pt",
#     target_dict_path: Path = "/path/to/dict.ltr.txt",
#     port: int = int(os.environ.get("ASR_RPYC_PORT", "8044")),
# ):
#     w2vasr = Wav2Vec2TransformersASR(ctc_path, w2v_path, target_dict_path)
#     service = ASRService(w2vasr)
#     logging.basicConfig(
#         level=logging.INFO,
#         format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
#     )
#     logging.info("starting asr server...")
#     t = ThreadedServer(service, port=port)
#     t.start()


@app.command()
def rpyc_dir(
    model_dir: Path, port: int = int(os.environ.get("ASR_RPYC_PORT", "8044"))
):
    typer.echo("loading asr model...")
    w2vasr = Wav2Vec2TransformersASR(model_dir)
    typer.echo("loaded asr model")
    service = ASRService(w2vasr)

    typer.echo(f"serving asr on :{port}...")
    t = ThreadedServer(service, port=port)
    t.start()


def main():
    app()


if __name__ == "__main__":
    main()

import os
import logging
from pathlib import Path

from rpyc.utils.server import ThreadedServer
import typer

# from .asr import JasperASR
from ...utils.serve import ASRService
from plume.utils import lazy_callable

JasperASR = lazy_callable('plume.models.jasper.asr.JasperASR')

app = typer.Typer()


@app.command()
def rpyc(
    encoder_path: Path = "/path/to/encoder.pt",
    decoder_path: Path = "/path/to/decoder.pt",
    model_yaml_path: Path = "/path/to/model.yaml",
    port: int = int(os.environ.get("ASR_RPYC_PORT", "8044")),
):
    for p in [encoder_path, decoder_path, model_yaml_path]:
        if not p.exists():
            logging.info(f"{p} doesn't exists")
            return
    asr = JasperASR(str(model_yaml_path), str(encoder_path), str(decoder_path))
    service = ASRService(asr)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logging.info("starting asr server...")
    t = ThreadedServer(service, port=port)
    t.start()


@app.command()
def rpyc_dir(model_dir: Path, port: int = int(os.environ.get("ASR_RPYC_PORT", "8044"))):
    encoder_path = model_dir / Path("decoder.pt")
    decoder_path = model_dir / Path("encoder.pt")
    model_yaml_path = model_dir / Path("model.yaml")
    rpyc(encoder_path, decoder_path, model_yaml_path, port)


def main():
    app()


if __name__ == "__main__":
    main()

import os
import logging
from pathlib import Path

# from rpyc.utils.server import ThreadedServer
import typer

from ...utils.serve import ASRService
from plume.utils import lazy_callable
# from .asr import Wav2Vec2ASR

ThreadedServer = lazy_callable('rpyc.utils.server.ThreadedServer')
Wav2Vec2ASR = lazy_callable('plume.models.wav2vec2.asr.Wav2Vec2ASR')

app = typer.Typer()


@app.command()
def rpyc(
    w2v_path: Path = "/path/to/base.pt",
    ctc_path: Path = "/path/to/ctc.pt",
    target_dict_path: Path = "/path/to/dict.ltr.txt",
    port: int = int(os.environ.get("ASR_RPYC_PORT", "8044")),
):
    for p in [w2v_path, ctc_path, target_dict_path]:
        if not p.exists():
            logging.info(f"{p} doesn't exists")
            return
    w2vasr = Wav2Vec2ASR(str(ctc_path), str(w2v_path), str(target_dict_path))
    service = ASRService(w2vasr)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logging.info("starting asr server...")
    t = ThreadedServer(service, port=port)
    t.start()


@app.command()
def rpyc_dir(model_dir: Path, port: int = int(os.environ.get("ASR_RPYC_PORT", "8044"))):
    ctc_path = model_dir / Path("ctc.pt")
    w2v_path = model_dir / Path("base.pt")
    target_dict_path = model_dir / Path("dict.ltr.txt")
    rpyc(w2v_path, ctc_path, target_dict_path, port)


def main():
    app()


if __name__ == "__main__":
    main()

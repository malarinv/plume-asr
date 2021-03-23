import os
import logging
from io import BytesIO
from pathlib import Path
from functools import lru_cache

import typer
# import rpyc

# from tqdm import tqdm
# from pydub.silence import split_on_silence
from plume.utils import lazy_module, lazy_callable

rpyc = lazy_module('rpyc')
pydub = lazy_module('pydub')
split_on_silence = lazy_callable('pydub.silence.split_on_silence')

app = typer.Typer()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


ASR_RPYC_HOST = os.environ.get("JASR_RPYC_HOST", "localhost")
ASR_RPYC_PORT = int(os.environ.get("ASR_RPYC_PORT", "8044"))

TRITON_ASR_MODEL = os.environ.get("TRITON_ASR_MODEL", "slu_wav2vec2")

TRITON_GRPC_ASR_HOST = os.environ.get("TRITON_GRPC_ASR_HOST", "localhost")
TRITON_GRPC_ASR_PORT = int(os.environ.get("TRITON_GRPC_ASR_PORT", "8001"))


@lru_cache()
def transcribe_rpyc_gen(asr_host=ASR_RPYC_HOST, asr_port=ASR_RPYC_PORT):
    logger.info(f"connecting to asr server at {asr_host}:{asr_port}")
    try:
        asr = rpyc.connect(asr_host, asr_port).root
        logger.info(f"connected to asr server successfully")
    except ConnectionRefusedError:
        raise Exception("env-var JASPER_ASR_RPYC_HOST invalid")

    def audio_prep(aud_seg):
        asr_seg = aud_seg.set_channels(1).set_sample_width(2).set_frame_rate(16000)
        return asr_seg

    return asr.transcribe, audio_prep


def triton_transcribe_grpc_gen(
    asr_host=TRITON_GRPC_ASR_HOST,
    asr_port=TRITON_GRPC_ASR_PORT,
    asr_model=TRITON_ASR_MODEL,
    method="chunked",
    chunk_msec=5000,
    sil_msec=500,
    # overlap=False,
    sep=" ",
):
    from tritonclient.utils import np_to_triton_dtype
    import tritonclient.grpc as grpcclient
    import numpy as np

    sup_meth = ["chunked", "silence", "whole"]
    if method not in sup_meth:
        meths = "|".join(sup_meth)
        raise Exception(f"unsupported method {method}. pick one of {meths}")

    client = grpcclient.InferenceServerClient(f"{asr_host}:{asr_port}")

    def transcriber(aud_seg):
        af = BytesIO()
        aud_seg.export(af, format="wav")
        input_audio_bytes = af.getvalue()
        input_audio_data = np.array([input_audio_bytes])
        inputs = [
            grpcclient.InferInput(
                "INPUT_AUDIO",
                input_audio_data.shape,
                np_to_triton_dtype(input_audio_data.dtype),
            )
        ]
        inputs[0].set_data_from_numpy(input_audio_data)
        outputs = [grpcclient.InferRequestedOutput("OUTPUT_TEXT")]
        response = client.infer(asr_model, inputs, request_id=str(1), outputs=outputs)
        transcript = response.as_numpy("OUTPUT_TEXT")[0]
        return transcript.decode("utf-8")

    def chunked_transcriber(aud_seg):
        if method == "silence":
            sil_chunks = split_on_silence(
                aud_seg,
                min_silence_len=sil_msec,
                silence_thresh=-50,
                keep_silence=500,
            )
            chunks = [sc for c in sil_chunks for sc in c[::chunk_msec]]
        else:
            chunks = aud_seg[::chunk_msec]
        # if overlap:
        #     chunks = [
        #         aud_seg[start, end]
        #         for start, end in range(0, int(aud_seg.duration_seconds * 1000, 1000))
        #     ]
        #     pass
        transcript_list = []
        sil_pad = pydub.AudioSegment.silent(duration=sil_msec)
        for seg in chunks:
            t_seg = sil_pad + seg + sil_pad
            c_transcript = transcriber(t_seg)
            transcript_list.append(c_transcript)
        transcript = sep.join(transcript_list)
        return transcript

    def audio_prep(aud_seg):
        asr_seg = aud_seg.set_channels(1).set_sample_width(2).set_frame_rate(16000)
        return asr_seg

    whole_transcriber = transcriber if method == "whole" else chunked_transcriber
    return whole_transcriber, audio_prep


@app.command()
def file(audio_file: Path, write_file: bool = False, chunked=True):
    aseg = pydub.AudioSegment.from_file(audio_file)
    transcriber, prep = triton_transcribe_grpc_gen()
    transcription = transcriber(prep(aseg))

    typer.echo(transcription)
    if write_file:
        tscript_file_path = audio_file.with_suffix(".txt")
        with open(tscript_file_path, "w") as tf:
            tf.write(transcription)


@app.command()
def benchmark(audio_file: Path):
    transcriber, audio_prep = transcribe_rpyc_gen()
    file_seg = pydub.AudioSegment.from_file(audio_file)
    aud_seg = audio_prep(file_seg)

    def timeinfo():
        from timeit import Timer

        timer = Timer(lambda: transcriber(aud_seg))
        number = 100
        repeat = 10
        time_taken = timer.repeat(repeat, number=number)
        best = min(time_taken) * 1000 / number
        print(f"{number} loops, best of {repeat}: {best:.3f} msec per loop")

    timeinfo()
    import time

    time.sleep(5)

    transcriber, audio_prep = triton_transcribe_grpc_gen()
    aud_seg = audio_prep(file_seg)

    def timeinfo():
        from timeit import Timer

        timer = Timer(lambda: transcriber(aud_seg))
        number = 100
        repeat = 10
        time_taken = timer.repeat(repeat, number=number)
        best = min(time_taken) * 1000 / number
        print(f"{number} loops, best of {repeat}: {best:.3f} msec per loop")

    timeinfo()


def main():
    app()


if __name__ == "__main__":
    main()

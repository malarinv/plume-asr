from pathlib import Path
import typer
from tqdm import tqdm
# import pandas as pd

from plume.utils import (
    asr_manifest_reader,
    discard_except_digits,
    replace_digit_symbol,
    lazy_module
    # run_shell,
)
from ...utils.transcribe import triton_transcribe_grpc_gen, transcribe_rpyc_gen

pd = lazy_module('pandas')
app = typer.Typer()


@app.command()
def manifest(manifest_file: Path, result_file: Path = "results.csv", rpyc: bool = False):
    from pydub import AudioSegment

    host = "localhost"
    port = 8044
    if rpyc:
        transcriber, audio_prep = transcribe_rpyc_gen(host, port)
    else:
        transcriber, audio_prep = triton_transcribe_grpc_gen(host, port, method='whole')
    result_path = manifest_file.parent / result_file
    manifest_list = list(asr_manifest_reader(manifest_file))

    def compute_frame(d):
        audio_file = d["audio_path"]
        orig_text = d["text"]
        orig_num = discard_except_digits(replace_digit_symbol(orig_text))
        aud_seg = AudioSegment.from_file(audio_file)
        t_audio = audio_prep(aud_seg)
        asr_text = transcriber(t_audio)
        asr_num = discard_except_digits(replace_digit_symbol(asr_text))
        return {
            "audio_file": audio_file,
            "asr_text": asr_text,
            "asr_num": asr_num,
            "orig_text": orig_text,
            "orig_num": orig_num,
            "asr_match": orig_num == asr_num,
        }

    # df_data = parallel_apply(compute_frame, manifest_list)
    df_data = map(compute_frame, tqdm(manifest_list))
    df = pd.DataFrame(df_data)
    df.to_csv(result_path)

from pathlib import Path
import shutil

import typer
from plume.utils.lazy_import import lazy_module
from plume.utils.tts import GoogleTTS
from plume.utils.transcribe import triton_transcribe_grpc_gen
from plume.utils.manifest import asr_manifest_writer

pydub = lazy_module("pydub")
app = typer.Typer()


@app.callback()
def generate():
    """
    generate sub commands
    """


@app.command()
def tts_dataset(dest_path: Path):
    tts = GoogleTTS()
    pass


@app.command()
def asr_dataset(audio_dir: Path, out_dir: Path, model="slu_num_wav2vec2"):
    out_wav_dir = out_dir / "wavs"
    out_wav_dir.mkdir(exist_ok=True, parents=True)

    def data_gen():
        aud_files = list(audio_dir.glob("*.mp3")) + list(
            audio_dir.glob("*.wav")
        )
        transcriber, prep = triton_transcribe_grpc_gen(
            asr_model=model, method="whole", append_raw=True
        )
        for af in aud_files:
            out_af = out_wav_dir / af.name
            audio_af = out_af.relative_to(out_dir)
            shutil.copy2(af, out_af)
            aud_seg = pydub.AudioSegment.from_file(out_af)
            t_seg = prep(aud_seg)
            transcript = transcriber(t_seg)
            # [digit_tscript, raw_tscript] = transcript.split("|")
            yield {
                "audio_filepath": str(audio_af),
                "duration": aud_seg.duration_seconds,
                "text": transcript,
            }

    asr_manifest_writer(out_dir / 'manifest.json', data_gen())

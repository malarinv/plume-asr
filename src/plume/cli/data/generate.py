from pathlib import Path
import shutil
from tqdm import tqdm
import typer
from plume.utils.lazy_import import lazy_module
from plume.utils.tts import GoogleTTS
from plume.utils.transcribe import (
    triton_transcribe_grpc_gen,
    chunk_transcribe_meta_gen,
    transcribe_rpyc_gen,
)
from plume.utils.manifest import asr_manifest_writer
from plume.utils.diarize import diarize_audio_gen
from plume.utils.extended_path import ExtendedPath

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

    asr_manifest_writer(out_dir / "manifest.json", data_gen())


@app.command()
def mono_diarize_asr_dataset(audio_dir: Path, out_dir: Path):
    out_wav_dir = out_dir / "wavs"
    out_wav_dir.mkdir(exist_ok=True, parents=True)
    diarize_audio = diarize_audio_gen()

    def data_gen():
        aud_files = list(audio_dir.glob("*/*.mp3")) + list(
            audio_dir.glob("*/*.wav")
        )
        diameta = ExtendedPath(out_dir / "diameta.json")
        base_transcriber, base_prep = transcribe_rpyc_gen()
        transcriber, prep = chunk_transcribe_meta_gen(
            base_transcriber, base_prep, method="chunked"
        )

        diametadata = []
        for af in tqdm(aud_files):
            try:
                # raise Exception("Test")
                for dres in diarize_audio(af):
                    sample_fname = dres.pop("sample_fname")
                    out_af = out_wav_dir / sample_fname
                    wav_bytes = dres.pop("wav")
                    out_af.write_bytes(wav_bytes)
                    audio_af = out_af.relative_to(out_dir)
                    aud_seg = dres.pop("wavseg")
                    t_seg = prep(aud_seg)
                    transcript = transcriber(t_seg)
                    diametadata.append(dres)
                    yield {
                        "audio_filepath": str(audio_af),
                        "duration": aud_seg.duration_seconds,
                        "text": transcript,
                    }
            except Exception as e:
                print(f'error diariziaing/trascribing {af} - {e}')
        diameta.write_json(diametadata)

    asr_manifest_writer(out_dir / "manifest.json", data_gen())

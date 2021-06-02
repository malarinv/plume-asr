from pathlib import Path
# from IPython import display
import io
import shutil

import typer
from plume.utils import lazy_module

from .tts import GoogleTTS

display = lazy_module('IPython.display')
pydub = lazy_module('pydub')
requests = lazy_module('requests')

app = typer.Typer()

# Start gentle with following command
# docker run --rm -d  --name gentle_service -p 8765:8765/tcp lowerquality/gentle


def gentle_aligner(service_uri, wav_data, utter_text):
    # service_uri= "http://52.41.161.36:8765/transcriptions"
    wav_f = io.BytesIO(wav_data)
    wav_seg = pydub.AudioSegment.from_file(wav_f)

    mp3_f = io.BytesIO()
    wav_seg.export(mp3_f, format="mp3")
    mp3_f.seek(0)
    params = (("async", "false"),)
    files = {
        "audio": ("audio.mp3", mp3_f),
        "transcript": ("words.txt", io.BytesIO(utter_text.encode("utf-8"))),
    }

    response = requests.post(service_uri, params=params, files=files)
    print(f"Time duration of audio {wav_seg.duration_seconds}")
    print(f"Time taken to align: {response.elapsed}s")
    return wav_seg, response.json()


def gentle_align_iter(service_uri, wav_data, utter_text):
    wav_seg, response = gentle_aligner(service_uri, wav_data, utter_text)
    for span in response:
        word_seg = wav_seg[int(span["start"] * 1000) : int(span["end"] * 1000)]
        word = span["word"]
        yield (word, word_seg)


def tts_jupyter():
    google_voices = GoogleTTS.voice_list()
    gtts = GoogleTTS()
    # google_voices[4]
    us_voice = [v for v in google_voices if v["language"] == "en-US"][0]
    utter_text = (
        "I would like to align the audio segments based on word level timestamps"
    )
    wav_data = gtts.text_to_speech(text=utter_text, params=us_voice)
    for word, seg in gentle_align_iter(wav_data, utter_text):
        print(word)
        display.display(seg)


@app.command()
def gentle_preview(
    audio_path: Path,
    transcript_path: Path,
    service_uri="http://101.53.142.218:8765/transcriptions",
    gent_preview_dir="./gentle_preview",
):
    from . import ExtendedPath

    pkg_gentle_dir = Path(__file__).parent / 'gentle_preview'

    shutil.copytree(str(pkg_gentle_dir), str(gent_preview_dir))
    ab = audio_path.read_bytes()
    tt = transcript_path.read_text()
    audio, alignment = gentle_aligner(service_uri, ab, tt)
    audio.export(gent_preview_dir / Path("a.wav"), format="wav")
    alignment["status"] = "OK"
    ExtendedPath(gent_preview_dir / Path("status.json")).write_json(alignment)


def main():
    app()


if __name__ == "__main__":
    main()

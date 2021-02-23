from pathlib import Path
from .tts import GoogleTTS
# from IPython import display
import requests
import io
import typer

from plume.utils import lazy_module

display = lazy_module('IPython.display')
pydub = lazy_module('pydub')

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
def cut(audio_path: Path, transcript_path: Path, out_dir: Path = "/tmp"):
    from . import ExtendedPath
    import datetime
    import re

    aud_seg = pydub.AudioSegment.from_file(audio_path)
    aud_seg[: 15 * 60 * 1000].export(out_dir / Path("audio.mp3"), format="mp3")
    tscript_json = ExtendedPath(transcript_path).read_json()

    def time_to_msecs(time_str):
        return (
            datetime.datetime.strptime(time_str, "%H:%M:%S,%f")
            - datetime.datetime(1900, 1, 1)
        ).total_seconds() * 1000

    tscript_words = []
    broken = False
    for m in tscript_json["monologues"]:
        # tscript_words.append("|")
        for e in m["elements"]:
            if e["type"] == "text":
                text = e["value"]
                text = re.sub(r"\[.*\]", "", text)
                text = re.sub(r"\(.*\)", "", text)
                tscript_words.append(text)
            if "timestamp" in e and time_to_msecs(e["timestamp"]) >= 15 * 60 * 1000:
                broken = True
                break
        if broken:
            break
    (out_dir / Path("words.txt")).write_text("".join(tscript_words))


@app.command()
def gentle_preview(
    audio_path: Path,
    transcript_path: Path,
    service_uri="http://101.53.142.218:8765/transcriptions",
    gent_preview_dir="../gentle_preview",
):
    from . import ExtendedPath

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

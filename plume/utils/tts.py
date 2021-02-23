from logging import getLogger
from plume.utils import lazy_module


from pathlib import Path

import typer

# from google.cloud import texttospeech
texttospeech = lazy_module('google.cloud.texttospeech')

LOGGER = getLogger("googletts")

app = typer.Typer()


class GoogleTTS(object):
    def __init__(self):
        self.client = texttospeech.TextToSpeechClient()

    def text_to_speech(self, text: str, params: dict) -> bytes:
        tts_input = texttospeech.types.SynthesisInput(text=text)
        voice = texttospeech.types.VoiceSelectionParams(
            language_code=params["language"], name=params["name"]
        )
        audio_config = texttospeech.types.AudioConfig(
            audio_encoding=texttospeech.enums.AudioEncoding.LINEAR16,
            sample_rate_hertz=params["sample_rate"],
        )
        response = self.client.synthesize_speech(tts_input, voice, audio_config)
        audio_content = response.audio_content
        return audio_content

    def ssml_to_speech(self, text: str, params: dict) -> bytes:
        tts_input = texttospeech.types.SynthesisInput(ssml=text)
        voice = texttospeech.types.VoiceSelectionParams(
            language_code=params["language"], name=params["name"]
        )
        audio_config = texttospeech.types.AudioConfig(
            audio_encoding=texttospeech.enums.AudioEncoding.LINEAR16,
            sample_rate_hertz=params["sample_rate"],
        )
        response = self.client.synthesize_speech(tts_input, voice, audio_config)
        audio_content = response.audio_content
        return audio_content

    @classmethod
    def voice_list(cls):
        """Lists the available voices."""

        client = cls().client

        # Performs the list voices request
        voices = client.list_voices()
        results = []
        for voice in voices.voices:
            supported_eng_langs = [
                lang for lang in voice.language_codes if lang[:2] == "en"
            ]
            if len(supported_eng_langs) > 0:
                lang = ",".join(supported_eng_langs)
            else:
                continue

            ssml_gender = texttospeech.enums.SsmlVoiceGender(voice.ssml_gender)
            results.append(
                {
                    "name": voice.name,
                    "language": lang,
                    "gender": ssml_gender.name,
                    "engine": "wavenet" if "Wav" in voice.name else "standard",
                    "sample_rate": voice.natural_sample_rate_hertz,
                }
            )
        return results


@app.command()
def generate_audio_file(text, dest_path: Path = "./tts_audio.wav", voice="en-US-Wavenet-D"):
    tts = GoogleTTS()
    selected_voice = [v for v in tts.voice_list() if v["name"] == voice][0]
    wav_data = tts.text_to_speech(text, selected_voice)
    with dest_path.open("wb") as wf:
        wf.write(wav_data)


def main():
    app()


if __name__ == "__main__":
    main()

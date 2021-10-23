from pathlib import Path
from plume.utils import lazy_module
from plume.utils.audio import audio_seg_to_wav_bytes

pydub = lazy_module('pydub')
torch = lazy_module('torch')


def transform_audio(file_location, path_to_save):
    audio_seg = (
        pydub.AudioSegment.from_file(file_location)
        .set_frame_rate(16000)
        .set_sample_width(2)
    )
    audio_seg.export(path_to_save, format="wav")


def gen_diarizer():
    pipeline = torch.hub.load("pyannote/pyannote-audio", "dia")

    def _diarizer(audio_path):
        return pipeline({"audio": audio_path})

    return _diarizer


# base_transcriber, base_prep = transcribe_rpyc_gen()
# transcriber, prep = chunk_transcribe_meta_gen(
#     base_transcriber, base_prep, method="chunked")

# diarizer = gen_diarizer()


def diarize_audio_gen():
    diarizer = gen_diarizer()

    def _diarize_audio(audio_path: Path):
        aseg = (
            pydub.AudioSegment.from_file(audio_path)
            .set_frame_rate(16000)
            .set_sample_width(2)
            .set_channels(1)
        )
        aseg.export("/tmp/temp.wav", format="wav")
        diarization = diarizer("/tmp/temp.wav")
        for n, (turn, _, speaker) in enumerate(
            diarization.itertracks(yield_label=True)
        ):
            # speaker_label = "Agent" if speaker == "B" else "Customer"
            turn_seg = aseg[turn.start * 1000 : turn.end * 1000]
            sample_fname = (
                audio_path.stem + "_" + str(n) + ".wav"
            )
            yield {
                "speaker": speaker,
                "wav": audio_seg_to_wav_bytes(turn_seg),
                "wavseg": turn_seg,
                "start": turn.start,
                "end": turn.end,
                "turnidx": n,
                "filename": audio_path.name,
                "sample_fname": sample_fname
            }

    return _diarize_audio

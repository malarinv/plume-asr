import logging
import asyncio
import argparse
from pathlib import Path

import webrtcvad
import pydub
from pydub.playback import play
from pydub.utils import make_chunks


DEFAULT_CHUNK_DUR = 20

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def is_frame_voice(vad, seg, chunk_dur):
    return (
        True
        if (
            seg.duration_seconds == chunk_dur / 1000
            and vad.is_speech(seg.raw_data, seg.frame_rate)
        )
        else False
    )


class VADFilterAudio(object):
    """docstring for VADFilterAudio."""

    def __init__(self, chunk_dur=DEFAULT_CHUNK_DUR):
        super(VADFilterAudio, self).__init__()
        self.chunk_dur = chunk_dur
        self.vad = webrtcvad.Vad()

    def filter_segment(self, wav_seg):
        chunks = make_chunks(wav_seg, self.chunk_dur)
        speech_buffer = b""

        for i, c in enumerate(chunks[:-1]):
            voice_frame = is_frame_voice(self.vad, c, self.chunk_dur)
            if voice_frame:
                speech_buffer += c.raw_data
        filtered_seg = pydub.AudioSegment(
            data=speech_buffer,
            frame_rate=wav_seg.frame_rate,
            channels=wav_seg.channels,
            sample_width=wav_seg.sample_width,
        )
        return filtered_seg


class VADUtterance(object):
    """docstring for VADUtterance."""

    def __init__(
        self,
        max_silence=500,
        min_utterance=280,
        max_utterance=20000,
        chunk_dur=DEFAULT_CHUNK_DUR,
        start_cycles=3,
    ):
        super(VADUtterance, self).__init__()
        self.vad = webrtcvad.Vad()
        self.chunk_dur = chunk_dur
        # duration in millisecs
        self.max_sil = max_silence
        self.min_utt = min_utterance
        self.max_utt = max_utterance
        self.speech_start = start_cycles * chunk_dur

    def __repr__(self):
        return f"VAD(max_silence={self.max_sil},min_utterance:{self.min_utt},max_utterance:{self.max_utt})"

    async def stream_utterance(self, audio_stream):
        silence_buffer = pydub.AudioSegment.empty()
        voice_buffer = pydub.AudioSegment.empty()
        silence_threshold = False
        async for c in audio_stream:
            voice_frame = is_frame_voice(self.vad, c, self.chunk_dur)
            logger.debug(f"is audio stream voice? {voice_frame}")
            if voice_frame:
                silence_threshold = False
                voice_buffer += c
                silence_buffer = pydub.AudioSegment.empty()
            else:
                silence_buffer += c
            voc_dur = voice_buffer.duration_seconds * 1000
            sil_dur = silence_buffer.duration_seconds * 1000

            if voc_dur >= self.max_utt:
                logger.info(
                    f"detected voice overflow: voice duration {voice_buffer.duration_seconds}"
                )
                yield voice_buffer
                voice_buffer = pydub.AudioSegment.empty()

            if sil_dur >= self.max_sil:
                if voc_dur >= self.min_utt:
                    logger.info(
                        f"detected silence: voice duration {voice_buffer.duration_seconds}"
                    )
                    yield voice_buffer
                voice_buffer = pydub.AudioSegment.empty()
                # ignore/clear voice if silence reached threshold or indent the statement
                if not silence_threshold:
                    silence_threshold = True

        if voice_buffer:
            yield voice_buffer

    async def stream_events(self, audio_stream):
        """
        yields 0, voice_buffer for SpeechBuffer
        yields 1, None for StartedSpeaking
        yields 2, None for StoppedSpeaking
        yields 4, audio_stream
        """
        silence_buffer = pydub.AudioSegment.empty()
        voice_buffer = pydub.AudioSegment.empty()
        silence_threshold, started_speaking = False, False
        async for c in audio_stream:
            # yield (4, c)
            voice_frame = is_frame_voice(self.vad, c, self.chunk_dur)
            logger.debug(f"is audio stream voice? {voice_frame}")
            if voice_frame:
                silence_threshold = False
                voice_buffer += c
                silence_buffer = pydub.AudioSegment.empty()
            else:
                silence_buffer += c
            voc_dur = voice_buffer.duration_seconds * 1000
            sil_dur = silence_buffer.duration_seconds * 1000

            if voc_dur >= self.speech_start and not started_speaking:
                started_speaking = True
                yield (1, None)

            if voc_dur >= self.max_utt:
                logger.info(
                    f"detected voice overflow: voice duration {voice_buffer.duration_seconds}"
                )
                yield (0, voice_buffer)
                voice_buffer = pydub.AudioSegment.empty()
                started_speaking = False

            if sil_dur >= self.max_sil:
                if voc_dur >= self.min_utt:
                    logger.info(
                        f"detected silence: voice duration {voice_buffer.duration_seconds}"
                    )
                    yield (0, voice_buffer)
                voice_buffer = pydub.AudioSegment.empty()
                started_speaking = False
                # ignore/clear voice if silence reached threshold or indent the statement
                if not silence_threshold:
                    silence_threshold = True
                    yield (2, None)

        if voice_buffer:
            yield (0, voice_buffer)

    @classmethod
    async def stream_utterance_file(cls, audio_file):
        async def stream_gen():
            audio_seg = pydub.AudioSegment.from_file(audio_file).set_frame_rate(32000)
            chunks = make_chunks(audio_seg, DEFAULT_CHUNK_DUR)
            for c in chunks:
                yield c

        va_ut = cls()
        buffer_src = va_ut.stream_utterance(stream_gen())
        async for buf in buffer_src:
            play(buf)
            await asyncio.sleep(1)


class VADStreamGen(object):
    """docstring for VADStreamGen."""

    def __init__(self, arg):
        super(VADStreamGen, self).__init__()
        self.arg = arg


def main():
    prog = Path(__file__).stem
    parser = argparse.ArgumentParser(prog=prog, description="transcribes audio file")
    parser.add_argument(
        "--audio_file",
        type=argparse.FileType("rb"),
        help="audio file to transcribe",
        default="./test_utter2.wav",
    )
    args = parser.parse_args()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(VADUtterance.stream_utterance_file(args.audio_file))


if __name__ == "__main__":
    main()

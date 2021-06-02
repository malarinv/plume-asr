import logging
from .lazy_import import lazy_module

webrtcvad = lazy_module("webrtcvad")
pydub = lazy_module("pydub")

DEFAULT_CHUNK_DUR = 30
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


class VADUtterance(object):
    """docstring for VADUtterance."""

    def __init__(
        self,
        max_silence=500,
        min_utterance=280,
        max_utterance=20000,
        chunk_dur=DEFAULT_CHUNK_DUR,
        start_cycles=3,
        aggression=1,
    ):
        super(VADUtterance, self).__init__()
        self.vad = webrtcvad.Vad(aggression)
        self.chunk_dur = chunk_dur
        # duration in millisecs
        self.max_sil = max_silence
        self.min_utt = min_utterance
        self.max_utt = max_utterance
        self.speech_start = start_cycles * chunk_dur

    def __repr__(self):
        return f"VAD(max_silence={self.max_sil},min_utterance:{self.min_utt},max_utterance:{self.max_utt})"

    def stream_segments(self, audio_seg):
        stream_seg = audio_seg.set_channels(1).set_sample_width(2).set_frame_rate(16000)
        silence_buffer = pydub.AudioSegment.empty()
        voice_buffer = pydub.AudioSegment.empty()
        silence_threshold = False
        for c in stream_seg[:: self.chunk_dur]:
            voice_frame = is_frame_voice(self.vad, c, self.chunk_dur)
            # logger.info(f"is audio stream voice? {voice_frame}")
            if voice_frame:
                silence_threshold = False
                voice_buffer += c
                silence_buffer = pydub.AudioSegment.empty()
            else:
                silence_buffer += c
            voc_dur = len(voice_buffer)
            sil_dur = len(silence_buffer)

            if voc_dur >= self.max_utt:
                # logger.info(
                #     f"detected voice overflow: voice duration {voice_buffer.duration_seconds}"
                # )
                yield voice_buffer
                voice_buffer = pydub.AudioSegment.empty()

            if sil_dur >= self.max_sil:
                if voc_dur >= self.min_utt:
                    # logger.info(
                    #     f"detected silence: voice duration {voice_buffer.duration_seconds}"
                    # )
                    yield voice_buffer
                voice_buffer = pydub.AudioSegment.empty()
                # ignore/clear voice if silence reached threshold or indent the statement
                if not silence_threshold:
                    silence_threshold = True

        # if voice_buffer:
        #     yield voice_buffer

        if self.min_utt < len(voice_buffer) < self.max_utt:
            yield voice_buffer

    # def stream_utterance(self, audio_stream):
    #     silence_buffer = pydub.AudioSegment.empty()
    #     voice_buffer = pydub.AudioSegment.empty()
    #     silence_threshold = False
    #     for avf in audio_stream:
    #         audio_bytes = avf.to_ndarray().tobytes()
    #         c = (
    #             pydub.AudioSegment(
    #                 data=audio_bytes,
    #                 frame_rate=avf.sample_rate,
    #                 channels=len(avf.layout.channels),
    #                 sample_width=avf.format.bytes,
    #             )
    #             .set_channels(1)
    #             .set_sample_width(2)
    #             .set_frame_rate(16000)
    #         )
    #         voice_frame = is_frame_voice(self.vad, c, self.chunk_dur)
    #         # logger.info(f"is audio stream voice? {voice_frame}")
    #         if voice_frame:
    #             silence_threshold = False
    #             voice_buffer += c
    #             silence_buffer = pydub.AudioSegment.empty()
    #         else:
    #             silence_buffer += c
    #         voc_dur = voice_buffer.duration_seconds * 1000
    #         sil_dur = silence_buffer.duration_seconds * 1000
    #
    #         if voc_dur >= self.max_utt:
    #             # logger.info(
    #             #     f"detected voice overflow: voice duration {voice_buffer.duration_seconds}"
    #             # )
    #             yield voice_buffer
    #             voice_buffer = pydub.AudioSegment.empty()
    #
    #         if sil_dur >= self.max_sil:
    #             if voc_dur >= self.min_utt:
    #                 # logger.info(
    #                 #     f"detected silence: voice duration {voice_buffer.duration_seconds}"
    #                 # )
    #                 yield voice_buffer
    #             voice_buffer = pydub.AudioSegment.empty()
    #             # ignore/clear voice if silence reached threshold or indent the statement
    #             if not silence_threshold:
    #                 silence_threshold = True
    #
    #     if voice_buffer:
    #         yield voice_buffer

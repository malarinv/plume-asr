from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# import soundfile as sf
from io import BytesIO
import torch

from plume.utils import lazy_module

sf = lazy_module("soundfile")


class Wav2Vec2TransformersASR(object):
    """docstring for Wav2Vec2TransformersASR."""

    def __init__(self, model_dir):
        super(Wav2Vec2TransformersASR, self).__init__()
        self.processor = Wav2Vec2Processor.from_pretrained(model_dir)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_dir)

    def transcribe(self, audio_data):
        aud_f = BytesIO(audio_data)
        # net_input = {}
        speech_data, _ = sf.read(aud_f)
        input_values = self.processor(
            speech_data, return_tensors="pt", padding="longest"
        ).input_values  # Batch size 1

        # retrieve logits
        logits = self.model(input_values).logits

        # take argmax and decode
        predicted_ids = torch.argmax(logits, dim=-1)

        transcription = self.processor.batch_decode(predicted_ids)[0]
        return transcription

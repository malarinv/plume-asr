from transformers import Wav2Vec2Tokenizer, Wav2Vec2ForCTC

# import soundfile as sf
from io import BytesIO
import torch

from plume.utils import lazy_module

sf = lazy_module("soundfile")


class Wav2Vec2TransformersASR(object):
    """docstring for Wav2Vec2TransformersASR."""

    def __init__(self, ctc_path, w2v_path, target_dict_path):
        super(Wav2Vec2TransformersASR, self).__init__()
        self.tokenizer = Wav2Vec2Tokenizer.from_pretrained(
            "facebook/wav2vec2-large-960h-lv60-self"
        )
        self.model = Wav2Vec2ForCTC.from_pretrained(
            "facebook/wav2vec2-large-960h-lv60-self"
        )

    def transcribe(self, audio_data):
        aud_f = BytesIO(audio_data)
        # net_input = {}
        speech_data, _ = sf.read(aud_f)
        input_values = self.tokenizer(
            speech_data, return_tensors="pt", padding="longest"
        ).input_values  # Batch size 1

        # retrieve logits
        logits = self.model(input_values).logits

        # take argmax and decode
        predicted_ids = torch.argmax(logits, dim=-1)

        transcription = self.tokenizer.batch_decode(predicted_ids)[0]
        return transcription

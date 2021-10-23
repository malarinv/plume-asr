from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# import soundfile as sf
from io import BytesIO
import torch

from plume.utils import lazy_module

sf = lazy_module("soundfile")


class Wav2Vec2TransformersASR(object):
    """docstring for Wav2Vec2TransformersASR."""

    def __init__(self, model_dir):
        # super(Wav2Vec2TransformersASR, self).__init__()
        self.device = "cuda:1" if torch.cuda.is_available() else "cpu"
        # sd = torch.load(
        #     model_dir / "pytorch_model.bin", map_location=self.device
        # )
        # self.processor = Wav2Vec2Processor.from_pretrained(
        #     model_dir, state_dict=sd
        # )
        # self.model = Wav2Vec2ForCTC.from_pretrained(model_dir, state_dict=sd).to(self.device)
        self.processor = Wav2Vec2Processor.from_pretrained(model_dir)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_dir).to(self.device)

    def transcribe(self, audio_data):
        aud_f = BytesIO(audio_data)
        # net_input = {}
        speech_data, _ = sf.read(aud_f)
        input_values = self.processor(
            speech_data,
            return_tensors="pt",
            padding="longest",
            sampling_rate=16000,
        ).input_values.to(
            self.device
        )  # Batch size 1

        # retrieve logits
        #print(f"audio:{speech_data.shape} processed:{input_values.shape}")
        logits = self.model(input_values).logits
        #print(f"logit shape:{logits.shape}")
        # take argmax and decode
        predicted_ids = torch.argmax(logits, dim=-1)
        #print(f"predicted_ids shape:{predicted_ids.shape}")
        transcription = self.processor.batch_decode(predicted_ids)[0]
        result = transcription.replace('<s>', '')
        return result

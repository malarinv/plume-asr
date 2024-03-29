from transformers import Wav2Vec2Tokenizer, Wav2Vec2ForCTC
from datasets import load_dataset
import soundfile as sf
import torch

# load model and tokenizer
tokenizer = Wav2Vec2Tokenizer.from_pretrained(
    "facebook/wav2vec2-large-960h-lv60-self"
)
model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-large-960h-lv60-self"
)


# define function to read in sound file
def map_to_array(batch):
    speech, _ = sf.read(batch["file"])
    batch["speech"] = speech
    return batch


# load dummy dataset and read soundfiles
def main():
    ds = load_dataset(
        "patrickvonplaten/librispeech_asr_dummy", "clean", split="validation"
    )
    ds = ds.map(map_to_array)

    # tokenize
    input_values = tokenizer(
        ds["speech"][:2], return_tensors="pt", padding="longest"
    ).input_values  # Batch size 1

    # retrieve logits
    logits = model(input_values).logits

    # take argmax and decode
    predicted_ids = torch.argmax(logits, dim=-1)

    transcription = tokenizer.batch_decode(predicted_ids)
    print(transcription)


if __name__ == "__main__":
    main()
